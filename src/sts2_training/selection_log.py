from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

SelectionEventLogger = Callable[[Mapping[str, Any]], None]

_LOG = logging.getLogger(__name__)
_ROOM_END_BOUNDARIES = frozenset(
    {
        "combat_terminal",
        "map_select",
        "room_end",
        "room_terminal",
        "run_terminal",
        "terminal",
    }
)


class SelectionAudit:
    """Keep the latest public Decision per Branch and emit one record per selection."""

    def __init__(self, logger: SelectionEventLogger | None) -> None:
        self._logger = logger
        self._decisions: dict[tuple[str, str], dict[str, Any]] = {}
        # Exact replay is tracked per logical selection, not merely per wire request.
        # A v0.7 emulate_actions batch intentionally gives every item the same request_id,
        # so branch_id is part of the identity to keep sibling items as independent
        # selections while still recognizing each item on an exact batch replay.
        self._selection_keys: set[tuple[str, str]] = set()

    def clear(self) -> None:
        self._decisions.clear()
        self._selection_keys.clear()

    def remember(self, response: Mapping[str, Any]) -> None:
        if self._logger is None:
            return
        instance_id = response.get("instance_id")
        branch_id = response.get("branch_id", "root")
        decision_point_id = response.get("decision_point_id")
        identifiers = (instance_id, branch_id, decision_point_id)
        if not all(isinstance(value, str) and value for value in identifiers):
            return
        if not isinstance(response.get("masked_emulator_dto"), Mapping):
            return
        self._decisions[(instance_id, branch_id)] = deepcopy(dict(response))

    def record_action(
        self,
        request: Mapping[str, Any],
        *,
        source_branch_id: str,
        result: Mapping[str, Any] | None,
        error: BaseException | None = None,
    ) -> None:
        if self._logger is None:
            return

        # Batch item audit records are normalized by the client, but keep this fallback
        # here so SelectionAudit itself is robust to an item-shaped request created by an
        # older v0.7 caller that omitted the operation field.
        event_request = dict(request)
        operation = event_request.get("operation")
        if not isinstance(operation, str) or not operation:
            operation = "emulate_actions"
            event_request["operation"] = operation

        received = self._decisions.get((str(event_request["instance_id"]), source_branch_id))
        if (
            received is not None
            and received.get("decision_point_id") != event_request["decision_point_id"]
        ):
            received = None

        request_id = event_request.get("request_id")
        branch_id = event_request.get("branch_id")
        logical_branch_id = (
            branch_id if isinstance(branch_id, str) and branch_id else source_branch_id
        )
        selection_key = (
            (request_id, logical_branch_id)
            if isinstance(request_id, str) and request_id and logical_branch_id
            else None
        )
        is_retry_of_selection = (
            selection_key is not None and selection_key in self._selection_keys
        )

        # A completion-uncertain action is recorded on its first attempt so external
        # cancellation and transport failures remain auditable. Exact replay is transport
        # recovery for the same logical selection. For emulate_actions the wire request_id
        # is shared by the whole batch, so the branch_id component above prevents sibling
        # items from being mistaken for recoveries of one another.
        event: dict[str, Any] = {
            "event": "selection_recovery" if is_retry_of_selection else "selection",
            "received": received,
            "request": event_request,
            "result": dict(result) if result is not None else None,
        }
        if operation == "commit_action" and result is not None:
            event.update(_root_outcomes(received, result))
        if error is not None:
            event["client_error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }

        try:
            self._logger(deepcopy(event))
        except Exception:  # noqa: BLE001 - audit failure must not alter gameplay
            _LOG.exception("selection logger failed")

        if not is_retry_of_selection and selection_key is not None:
            self._selection_keys.add(selection_key)

        successful = (
            error is None
            and result is not None
            and result.get("status") in {"completed", "partial"}
        )
        if successful:
            root_committed = (
                operation == "commit_action"
                and result.get("status") == "completed"
            )
            if root_committed:
                self.clear()
            self.remember(result)


def _masked_dto(response: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if response is None:
        return {}
    value = response.get("masked_emulator_dto")
    return value if isinstance(value, Mapping) else {}


def _root_outcomes(
    received: Mapping[str, Any] | None,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    before = _masked_dto(received)
    after = _masked_dto(result)
    before_boundary = before.get("boundary")
    after_boundary = after.get("boundary")
    outcomes: dict[str, Any] = {}

    if _room_completed(before_boundary, after_boundary):
        outcomes["room_result"] = {
            "boundary_before": before_boundary,
            "boundary_after": after_boundary,
            "room_context_before": before.get("room_context"),
            "room_context_after": after.get("room_context"),
        }
    if _is_run_terminal(after):
        run_result = after.get("run_result")
        if run_result is None:
            run_result = after.get("run_outcome")
        if run_result is None:
            run_result = {
                "boundary": after_boundary,
                "room_context": after.get("room_context"),
                "transition": after.get("transition"),
            }
        outcomes["run_result"] = run_result
    return outcomes


def _room_completed(before: Any, after: Any) -> bool:
    if not isinstance(before, str) or not isinstance(after, str):
        return False
    before, after = before.lower(), after.lower()
    return after in _ROOM_END_BOUNDARIES and before not in {after, "map_select"}


def _is_run_terminal(dto: Mapping[str, Any]) -> bool:
    boundary = dto.get("boundary")
    if isinstance(boundary, str) and boundary.lower() == "run_terminal":
        return True
    if dto.get("run_terminal") is True:
        return True
    if dto.get("run_result") is not None or dto.get("run_outcome") is not None:
        return True
    transition = dto.get("transition")
    kind = transition.get("kind") if isinstance(transition, Mapping) else None
    return isinstance(kind, str) and kind.lower() in {
        "run_completed",
        "run_end",
        "run_terminal",
    }


class JsonlSelectionLogger:
    """Write one flushed UTF-8 JSON object per audit event."""

    def __init__(
        self,
        path: str | Path,
        *,
        append: bool = True,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self._stream: TextIO | None = self.path.open(
            "a" if append else "w", encoding="utf-8", newline="\n"
        )

    def __call__(self, event: Mapping[str, Any]) -> None:
        timestamp = self._clock()
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        record = dict(event)
        record.setdefault(
            "logged_at",
            timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            if self._stream is None:
                raise RuntimeError("selection logger is closed")
            self._stream.write(f"{line}\n")
            self._stream.flush()

    def close(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None

    def __enter__(self) -> JsonlSelectionLogger:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
