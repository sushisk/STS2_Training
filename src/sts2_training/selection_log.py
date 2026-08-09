from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

SelectionEventLogger = Callable[[Mapping[str, Any]], None]
SelectionIdentity = tuple[str, str]

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


def _selection_identity(
    request: Mapping[str, Any],
    source_branch_id: str,
) -> SelectionIdentity | None:
    request_id = request.get("request_id")
    branch_id = request.get("branch_id")
    logical_branch_id = (
        branch_id if isinstance(branch_id, str) and branch_id else source_branch_id
    )
    if not isinstance(request_id, str) or not request_id:
        return None
    if not isinstance(logical_branch_id, str) or not logical_branch_id:
        return None
    return request_id, logical_branch_id


class SelectionAudit:
    """Keep the latest public Decision per Branch and emit one record per selection."""

    def __init__(self, logger: SelectionEventLogger | None) -> None:
        self._logger = logger
        self._decisions: dict[tuple[str, str], dict[str, Any]] = {}
        # The protocol is single-in-flight, so replay identity is needed only for the
        # current wire request. Keeping one request_id plus its Branch IDs bounds this
        # state by batch size rather than total speculative selections.
        self._selection_request_id: str | None = None
        self._selection_branch_ids: set[str] = set()
        # Whole Run's deployed resolve_action_id interprets root commit tokens as
        # positional ordinals even though its public legal_actions expose sparse opaque
        # action_id values. ApiContract supplies this instance-level compatibility fact
        # explicitly when start_instance is accepted so audit semantics never have to
        # infer it from an unrelated capability field.
        self._wire_commit_action_id_is_ordinal = False

    def _clear_transient(self) -> None:
        """Drop decision/replay state while preserving active-instance wire quirks."""
        self._decisions.clear()
        self._selection_request_id = None
        self._selection_branch_ids.clear()

    def clear(self) -> None:
        self._clear_transient()
        self._wire_commit_action_id_is_ordinal = False

    def forget(self, instance_id: str, branch_ids: Sequence[str]) -> None:
        """Drop cached Decisions for Branches that can no longer be selected."""
        for branch_id in branch_ids:
            self._decisions.pop((instance_id, branch_id), None)

    def remember(
        self,
        response: Mapping[str, Any],
        *,
        wire_commit_action_id_is_ordinal: bool | None = None,
    ) -> None:
        if self._logger is None:
            return
        if wire_commit_action_id_is_ordinal is not None:
            self._wire_commit_action_id_is_ordinal = wire_commit_action_id_is_ordinal
        instance_id = response.get("instance_id")
        branch_id = response.get("branch_id", "root")
        decision_point_id = response.get("decision_point_id")
        identifiers = (instance_id, branch_id, decision_point_id)
        if not all(isinstance(value, str) and value for value in identifiers):
            return
        if not isinstance(response.get("masked_emulator_dto"), Mapping):
            return
        self._decisions[(instance_id, branch_id)] = deepcopy(dict(response))

    def _is_recovery(self, identity: SelectionIdentity | None) -> bool:
        if identity is None:
            return False

        request_id, branch_id = identity
        if request_id != self._selection_request_id:
            self._selection_request_id = request_id
            self._selection_branch_ids.clear()
        return branch_id in self._selection_branch_ids

    def _remember_selection(self, identity: SelectionIdentity | None) -> None:
        if identity is None:
            return
        _, branch_id = identity
        self._selection_branch_ids.add(branch_id)

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
        # so SelectionAudit also handles item-shaped v0.7 requests without an operation.
        event_request = dict(request)
        operation = event_request.get("operation")
        if not isinstance(operation, str) or not operation:
            operation = "emulate_actions"
            event_request["operation"] = operation

        received = self._decisions.get(
            (str(event_request["instance_id"]), source_branch_id)
        )
        if (
            received is not None
            and received.get("decision_point_id") != event_request["decision_point_id"]
        ):
            received = None

        identity = _selection_identity(event_request, source_branch_id)
        is_retry_of_selection = self._is_recovery(identity)
        selected_action_id = _selected_public_action_id(
            received,
            event_request,
            wire_commit_action_id_is_ordinal=self._wire_commit_action_id_is_ordinal,
        )

        # Completion-uncertain first attempts remain auditable. Exact replay is recorded
        # as recovery for the same logical selection; Branch ID distinguishes siblings
        # that share one emulate_actions request_id.
        event: dict[str, Any] = {
            "event": "selection_recovery" if is_retry_of_selection else "selection",
            "received": received,
            "request": event_request,
            "result": dict(result) if result is not None else None,
            # Keep request.action_id byte-for-byte faithful to the wire request. This
            # separate field is the opaque public ID from received.legal_actions and is
            # therefore the stable label self-play consumers should train against.
            "selected_action_id": selected_action_id,
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

        if not is_retry_of_selection:
            self._remember_selection(identity)

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
                # Root commit invalidates all cached Branch Decisions, but the active
                # instance remains the same. Preserve its wire-ID compatibility mode for
                # the next Decision instead of resetting it as close/start would.
                self._clear_transient()
            remembered_result = dict(result)
            # emulate_actions item results omit instance_id; restore it from the
            # normalized request so next-depth audit lookup retains its observation.
            remembered_result.setdefault("instance_id", event_request.get("instance_id"))
            self.remember(remembered_result)


def _masked_dto(response: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if response is None:
        return {}
    value = response.get("masked_emulator_dto")
    return value if isinstance(value, Mapping) else {}


def _selected_public_action_id(
    received: Mapping[str, Any] | None,
    request: Mapping[str, Any],
    *,
    wire_commit_action_id_is_ordinal: bool,
) -> str | None:
    """Resolve the selected opaque public ID without rewriting the wire request.

    Normal Combat/emulation requests send the public action_id directly. The currently
    deployed Whole Run root commit path is the exception: Training temporarily sends the
    action's ordinal because RL's _View.resolve_action_id treats the token as a list
    index. Audit data must still label the selection with the sparse public ID exposed in
    received.masked_emulator_dto.legal_actions.
    """
    actions = _masked_dto(received).get("legal_actions")
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
        return None
    requested_action_id = request.get("action_id")
    if not isinstance(requested_action_id, str) or not requested_action_id:
        return None

    if request.get("operation") == "commit_action" and wire_commit_action_id_is_ordinal:
        try:
            index = int(requested_action_id)
        except ValueError:
            return None
        if index < 0 or index >= len(actions):
            return None
        selected = actions[index]
        if not isinstance(selected, Mapping):
            return None
        public_action_id = selected.get("action_id")
        return public_action_id if isinstance(public_action_id, str) and public_action_id else None

    for action in actions:
        if not isinstance(action, Mapping):
            continue
        public_action_id = action.get("action_id")
        if public_action_id == requested_action_id:
            return requested_action_id
    return None


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
            run_result = after.get("outcome")
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