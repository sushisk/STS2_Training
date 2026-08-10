from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any


class ReplayLogError(ValueError):
    """Raised when a JSONL replay file cannot be decoded safely."""


class EventStore:
    """Thread-safe, append-only event store shared by the runner and HTTP server."""

    def __init__(self, records: Iterable[Mapping[str, Any]] = ()) -> None:
        self._lock = threading.RLock()
        self._records: list[dict[str, Any]] = [deepcopy(dict(record)) for record in records]

    def append(self, record: Mapping[str, Any]) -> int:
        with self._lock:
            self._records.append(deepcopy(dict(record)))
            return len(self._records) - 1

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._records)

    def after(self, cursor: int) -> list[tuple[int, dict[str, Any]]]:
        """Return records whose zero-based index is greater than ``cursor``."""
        with self._lock:
            start = max(cursor + 1, 0)
            return [(index, deepcopy(record)) for index, record in enumerate(self._records[start:], start)]

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a selection log while preserving record order and actionable errors."""
    log_path = Path(path)
    records: list[dict[str, Any]] = []
    try:
        stream = log_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ReplayLogError(f"cannot open replay log {log_path}: {exc}") from exc

    with stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReplayLogError(
                    f"invalid JSON in {log_path} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise ReplayLogError(
                    f"record in {log_path} at line {line_number} must be a JSON object"
                )
            records.append(value)
    return records


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _masked_dto(container: Any) -> dict[str, Any]:
    value = _mapping(container).get("masked_emulator_dto")
    return dict(value) if isinstance(value, Mapping) else {}


def _selected_action(record: Mapping[str, Any], before: Mapping[str, Any]) -> dict[str, Any] | None:
    selected_id = record.get("selected_action_id")
    if not isinstance(selected_id, str) or not selected_id:
        request_id = _mapping(record.get("request")).get("action_id")
        selected_id = request_id if isinstance(request_id, str) else None
    if not selected_id:
        return None

    legal_actions = before.get("legal_actions")
    if not isinstance(legal_actions, list):
        return {"action_id": selected_id}
    for action in legal_actions:
        if isinstance(action, Mapping) and action.get("action_id") == selected_id:
            return deepcopy(dict(action))
    return {"action_id": selected_id}


def present_event(index: int, record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize audit records into a stable browser payload without losing raw data."""
    received = _mapping(record.get("received"))
    result = _mapping(record.get("result"))
    before = _masked_dto(received)
    after = _masked_dto(result)

    # Self-play terminal records are not SelectionAudit events, but they are useful
    # final replay frames and already carry the complete DTO.
    if not before and isinstance(record.get("final_dto"), Mapping):
        before = deepcopy(dict(record["final_dto"]))
    if not before and after:
        before = deepcopy(after)

    request = _mapping(record.get("request"))
    operation = request.get("operation")
    branch_id = request.get("branch_id") or received.get("branch_id") or result.get("branch_id")
    return {
        "index": index,
        "event": record.get("event", "unknown"),
        "logged_at": record.get("logged_at"),
        "operation": operation,
        "branch_id": branch_id,
        "decision_point_id": received.get("decision_point_id") or result.get("decision_point_id"),
        "selected_action_id": record.get("selected_action_id") or request.get("action_id"),
        "selected_action": _selected_action(record, before),
        "before": before,
        "after": after,
        "client_error": deepcopy(record.get("client_error")),
        "room_result": deepcopy(record.get("room_result")),
        "run_result": deepcopy(record.get("run_result")),
        "raw": deepcopy(dict(record)),
    }
