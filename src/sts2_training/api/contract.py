from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.selection_log import SelectionAudit, SelectionEventLogger

JsonObject = dict[str, Any]
SCHEMA_VERSION = "0.7"
ROOT_BRANCH_ID = "root"
ROOT_RNG_ID = 0
KNOWN_STATUSES = frozenset(
    {
        "completed",
        "partial",
        "queued",
        "running",
        "cancelled",
        "rejected",
        "faulted",
        "released",
    }
)
_BRANCH_BATCH_OPERATIONS = frozenset(
    {"cancel_branches", "release_branches", "get_branch_status"}
)
_BRANCH_STATUS_VALUES = frozenset(
    {"queued", "running", "completed", "cancelled", "faulted", "released"}
)
# emulate_actions is synchronous at the coordinator boundary: BranchManager.poll()
# returns only after every Branch dispatched for the batch is terminal. queued/running
# are therefore invalid per-item response states for this operation.
_EMULATE_ACTIONS_BRANCH_STATUSES = frozenset({"completed", "partial", "faulted"})
_COMBAT_TERMINAL_OUTCOMES = frozenset({"victory", "defeat"})
# Whole-run completion has a distinct victory token while sharing combat defeat semantics.
_RUN_TERMINAL_OUTCOMES = _COMBAT_TERMINAL_OUTCOMES | {"run_victory"}


class ApiProtocolError(RuntimeError):
    pass


class ApiOperationError(RuntimeError):
    def __init__(self, response: Mapping[str, Any]) -> None:
        self.response = dict(response)
        super().__init__(
            f"{response.get('operation')} returned {response.get('status')}: {response.get('error')}"
        )


class RequestRejectedError(ApiOperationError):
    pass


class RequestFaultedError(ApiOperationError):
    pass


class ApiContract:
    """DTO v0.7 construction, correlation, active-instance state, and selection audit."""

    def __init__(
        self,
        *,
        client_session_id: str | None = None,
        selection_logger: SelectionEventLogger | None = None,
    ) -> None:
        self._client_session_id = client_session_id or str(uuid.uuid4())
        self._validate_non_empty_str(self._client_session_id, "client_session_id")
        self._audit = SelectionAudit(selection_logger)
        self._instance_id: str | None = None
        self._instance_type: str | None = None
        self._pending_start_instance_type: str | None = None
        self._max_emulate_actions_items: int | None = None

    @property
    def client_session_id(self) -> str:
        return self._client_session_id

    @property
    def instance_id(self) -> str | None:
        return self._instance_id

    @property
    def instance_type(self) -> str | None:
        """Requested type for the active instance, when one has been accepted."""
        return self._instance_type

    @property
    def max_emulate_actions_items(self) -> int | None:
        """RL-published batch limit for the active instance, when supported."""
        return self._max_emulate_actions_items

    def _new_request(self, request_seq: int, operation: str, **fields: Any) -> JsonObject:
        if not isinstance(request_seq, int) or isinstance(request_seq, bool) or request_seq <= 0:
            raise ValueError("request_seq must be a positive integer")
        return {
            "schema_version": SCHEMA_VERSION,
            "client_session_id": self._client_session_id,
            "request_seq": request_seq,
            "request_id": f"{self._client_session_id}:{request_seq}",
            "operation": operation,
            **fields,
        }

    def _build_start_instance(self, request_seq: int, instance_config: Mapping[str, Any]) -> JsonObject:
        instance_type = instance_config.get("instance_type")
        self._pending_start_instance_type = (
            instance_type if isinstance(instance_type, str) and instance_type else None
        )
        return self._new_request(request_seq, "start_instance", instance_config=dict(instance_config))

    def _accept_start_instance(self, response: Mapping[str, Any]) -> str:
        self._require_status(response, {"completed"})
        instance_id = self._require_non_empty_str(response, "instance_id")
        max_items = response.get("max_emulate_actions_items")
        if max_items is not None and (
            isinstance(max_items, bool) or not isinstance(max_items, int) or max_items <= 0
        ):
            raise ApiProtocolError("max_emulate_actions_items must be a positive integer")
        self._instance_id = instance_id
        self._instance_type = self._pending_start_instance_type
        self._pending_start_instance_type = None
        self._max_emulate_actions_items = max_items
        self._audit.clear()
        self._audit.remember(
            response,
            wire_commit_action_id_is_ordinal=self._instance_type == "whole_run",
        )
        return instance_id

    def _build_get_decision(self, request_seq: int, instance_id: str, branch_id: str) -> JsonObject:
        self._validate_instance_id(instance_id)
        self._validate_non_empty_str(branch_id, "branch_id")
        return self._new_request(
            request_seq, "get_decision", instance_id=instance_id, branch_id=branch_id
        )

    def _accept_get_decision(self, response: JsonObject, branch_id: str) -> JsonObject:
        if response["status"] == "completed":
            self._require_response_match(response, "branch_id", branch_id)
            self._validate_decision_payload(response)
            self._audit.remember(response)
        return response

    def _build_commit_action(
        self,
        request_seq: int,
        instance_id: str,
        decision_point_id: str,
        action_id: str,
    ) -> JsonObject:
        self._validate_instance_id(instance_id)
        self._validate_non_empty_str(decision_point_id, "decision_point_id")
        self._validate_non_empty_str(action_id, "action_id")
        return self._new_request(
            request_seq,
            "commit_action",
            instance_id=instance_id,
            branch_id=ROOT_BRANCH_ID,
            rng_id=ROOT_RNG_ID,
            decision_point_id=decision_point_id,
            action_id=action_id,
        )

    def _build_emulate_action(
        self,
        request_seq: int,
        instance_id: str,
        parent_branch_id: str,
        branch_id: str,
        rng_id: int,
        decision_point_id: str,
        action_id: str,
        simulation_options: Mapping[str, Any] | None,
    ) -> JsonObject:
        self._validate_instance_id(instance_id)
        for value, name in (
            (parent_branch_id, "parent_branch_id"),
            (branch_id, "branch_id"),
            (decision_point_id, "decision_point_id"),
            (action_id, "action_id"),
        ):
            self._validate_non_empty_str(value, name)
        if branch_id == ROOT_BRANCH_ID:
            raise ValueError("emulate_action branch_id must not be 'root'")
        if not isinstance(rng_id, int) or isinstance(rng_id, bool) or rng_id <= 0:
            raise ValueError("emulate_action rng_id must be a positive integer")
        fields: JsonObject = {
            "instance_id": instance_id,
            "parent_branch_id": parent_branch_id,
            "branch_id": branch_id,
            "rng_id": rng_id,
            "decision_point_id": decision_point_id,
            "action_id": action_id,
        }
        if simulation_options is not None:
            self._validate_simulation_options(simulation_options)
            fields["simulation_options"] = dict(simulation_options)
        return self._new_request(request_seq, "emulate_action", **fields)

    def _build_emulate_actions(
        self,
        request_seq: int,
        instance_id: str,
        items: Sequence[Mapping[str, Any]],
        simulation_options: Mapping[str, Any] | None,
    ) -> JsonObject:
        """Build DTO v0.7's ``emulate_actions`` batch request.

        The wire contract requires every ``parent_branch_id`` to refer to a Branch that
        already exists when this batch request starts. Training cannot prove RL-side
        existence locally, but it can reject any parent that is created by another item
        in the same batch. Beam Search should send one frontier depth per batch.
        """
        self._validate_instance_id(instance_id)
        if isinstance(items, (str, bytes)):
            raise TypeError("items must be a sequence of item mappings")
        items = list(items)
        if not items:
            raise ValueError("emulate_actions items must not be empty")
        if (
            self._max_emulate_actions_items is not None
            and len(items) > self._max_emulate_actions_items
        ):
            raise ValueError(
                f"emulate_actions item count {len(items)} exceeds RL "
                f"max_emulate_actions_items={self._max_emulate_actions_items}; "
                "chunk the frontier into multiple requests"
            )

        seen_branch_ids: set[str] = set()
        normalized_items: list[JsonObject] = []
        for item in items:
            if not isinstance(item, Mapping):
                raise TypeError("each emulate_actions item must be a mapping")
            parent_branch_id = item["parent_branch_id"]
            branch_id = item["branch_id"]
            rng_id = item["rng_id"]
            decision_point_id = item["decision_point_id"]
            action_id = item["action_id"]
            for value, name in (
                (parent_branch_id, "parent_branch_id"),
                (branch_id, "branch_id"),
                (decision_point_id, "decision_point_id"),
                (action_id, "action_id"),
            ):
                self._validate_non_empty_str(value, name)
            if branch_id == ROOT_BRANCH_ID:
                raise ValueError("emulate_actions item branch_id must not be 'root'")
            if branch_id in seen_branch_ids:
                raise ValueError(f"emulate_actions item branch_id {branch_id!r} is duplicated")
            seen_branch_ids.add(branch_id)
            if not isinstance(rng_id, int) or isinstance(rng_id, bool) or rng_id <= 0:
                raise ValueError("emulate_actions item rng_id must be a positive integer")
            normalized_items.append(
                {
                    "parent_branch_id": parent_branch_id,
                    "branch_id": branch_id,
                    "rng_id": rng_id,
                    "decision_point_id": decision_point_id,
                    "action_id": action_id,
                }
            )

        batch_branch_ids = {item["branch_id"] for item in normalized_items}
        for item in normalized_items:
            parent_branch_id = item["parent_branch_id"]
            if parent_branch_id in batch_branch_ids:
                raise ValueError(
                    f"emulate_actions parent_branch_id {parent_branch_id!r} is created within the same batch"
                )

        fields: JsonObject = {"instance_id": instance_id, "items": normalized_items}
        if simulation_options is not None:
            self._validate_simulation_options(simulation_options)
            fields["simulation_options"] = dict(simulation_options)
        return self._new_request(request_seq, "emulate_actions", **fields)

    def _validate_emulate_actions_response(
        self,
        request: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> None:
        self._require_status(response, {"completed"})
        branch_results = response.get("branch_results")
        if not isinstance(branch_results, dict):
            raise ApiProtocolError("branch_results must be a dictionary")

        items = request.get("items")
        if not isinstance(items, list):
            raise ApiProtocolError("emulate_actions request is missing items")
        expected_items = {item["branch_id"]: item for item in items}
        if set(branch_results) != set(expected_items):
            raise ApiProtocolError("response branch_results keys do not match request items")

        for branch_id, branch_result in branch_results.items():
            if not isinstance(branch_result, Mapping):
                raise ApiProtocolError(f"branch_results[{branch_id!r}] must be a dictionary")
            status = branch_result.get("status")
            if status not in _EMULATE_ACTIONS_BRANCH_STATUSES:
                raise ApiProtocolError(f"invalid branch status for {branch_id!r}: {status!r}")

            expected = expected_items[branch_id]
            self._require_response_match(branch_result, "branch_id", branch_id)
            self._require_response_match(
                branch_result, "parent_branch_id", expected["parent_branch_id"]
            )
            self._require_response_match(branch_result, "rng_id", expected["rng_id"])

            if status in {"completed", "partial"}:
                self._validate_decision_payload(branch_result)
            elif status == "faulted":
                self._require_non_empty_str(branch_result, "error")

    def _build_branch_batch_operation(
        self,
        request_seq: int,
        operation: str,
        instance_id: str,
        branch_ids: Sequence[str],
    ) -> JsonObject:
        self._validate_instance_id(instance_id)
        normalized = self._normalize_branch_ids(branch_ids)
        return self._new_request(
            request_seq, operation, instance_id=instance_id, branch_ids=normalized
        )

    def _build_close_instance(self, request_seq: int, instance_id: str) -> JsonObject:
        self._validate_instance_id(instance_id)
        return self._new_request(request_seq, "close_instance", instance_id=instance_id)

    def _accept_close_instance(self, response: JsonObject) -> JsonObject:
        self._require_status(response, {"completed"})
        self._instance_id = None
        self._instance_type = None
        self._pending_start_instance_type = None
        self._max_emulate_actions_items = None
        self._audit.clear()
        return response

    def _validate_api_response(self, request: Mapping[str, Any], response: Any) -> JsonObject:
        if not isinstance(response, dict):
            raise ApiProtocolError("response must be a dictionary")
        if response.get("schema_version") != SCHEMA_VERSION:
            raise ApiProtocolError("schema_version does not match")
        for field in ("client_session_id", "request_seq", "request_id", "operation"):
            if response.get(field) != request.get(field):
                raise ApiProtocolError(f"response {field} does not match request")
        self._require_non_empty_str(response, "server_epoch")
        status = self._require_non_empty_str(response, "status")
        if status not in KNOWN_STATUSES:
            raise ApiProtocolError(f"unknown status: {status}")
        if request.get("instance_id") is not None:
            self._require_response_match(response, "instance_id", request["instance_id"])
        if status in {"rejected", "faulted"}:
            self._require_non_empty_str(response, "error")
        if status == "rejected":
            raise RequestRejectedError(response)
        if status == "faulted":
            raise RequestFaultedError(response)
        if request.get("operation") in _BRANCH_BATCH_OPERATIONS:
            self._validate_branch_batch_response(request, response)
        elif request.get("operation") == "emulate_actions":
            self._validate_emulate_actions_response(request, response)
        return dict(response)

    def _validate_branch_batch_response(
        self,
        request: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> None:
        self._require_status(response, {"completed"})
        branch_statuses = response.get("branch_statuses")
        if not isinstance(branch_statuses, dict):
            raise ApiProtocolError("branch_statuses must be a dictionary")

        branch_ids = request.get("branch_ids")
        if not isinstance(branch_ids, list):
            raise ApiProtocolError("branch batch request is missing branch_ids")
        if set(branch_statuses) != set(branch_ids):
            raise ApiProtocolError("response branch_statuses keys do not match request")

        operation = request.get("operation")
        if operation == "cancel_branches":
            allowed_statuses = frozenset({"cancelled"})
        elif operation == "release_branches":
            allowed_statuses = frozenset({"released"})
        else:
            allowed_statuses = _BRANCH_STATUS_VALUES

        for branch_id, branch_status in branch_statuses.items():
            if not isinstance(branch_status, str) or branch_status not in allowed_statuses:
                raise ApiProtocolError(
                    f"invalid branch status for {branch_id!r}: {branch_status!r}"
                )

        # Once RL has confirmed cancellation or release, the Branch can no longer be
        # selected as a parent. Retaining its deep-copied Decision payload would make
        # audit memory grow with the total number of speculative Branches ever created.
        if operation in {"cancel_branches", "release_branches"}:
            instance_id = request.get("instance_id")
            if isinstance(instance_id, str) and instance_id:
                self._audit.forget(instance_id, branch_ids)

    def _validate_selected_action_response(
        self, request: Mapping[str, Any], response: Mapping[str, Any]
    ) -> None:
        status = response["status"]
        if request["operation"] == "commit_action":
            if status == "completed":
                self._require_response_match(response, "branch_id", ROOT_BRANCH_ID)
                self._validate_decision_payload(response)
            return
        if status in {"completed", "partial", "queued", "running"}:
            self._require_response_match(response, "branch_id", request["branch_id"])
            if "parent_branch_id" in response:
                self._require_response_match(response, "parent_branch_id", request["parent_branch_id"])
            if "rng_id" in response:
                self._require_response_match(response, "rng_id", request["rng_id"])
        if status in {"completed", "partial"}:
            self._validate_decision_payload(response)

    def _record_selected_action(
        self,
        request: Mapping[str, Any],
        *,
        source_branch_id: str,
        result: Mapping[str, Any] | None,
        error: BaseException | None = None,
    ) -> None:
        self._audit.record_action(
            request, source_branch_id=source_branch_id, result=result, error=error
        )

    def _validate_instance_id(self, instance_id: str) -> None:
        self._validate_non_empty_str(instance_id, "instance_id")
        if self._instance_id is None:
            raise RuntimeError("client has no active instance")
        if instance_id != self._instance_id:
            raise ValueError("instance_id does not match the active client instance")

    @staticmethod
    def _validate_simulation_options(options: Mapping[str, Any]) -> None:
        if not isinstance(options, Mapping):
            raise TypeError("simulation_options must be a mapping")
        for key in ("max_depth", "max_steps", "max_time_ms", "max_hypotheses"):
            value = options.get(key)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"simulation_options.{key} must be a positive integer")
        stop_condition = options.get("stop_condition")
        if stop_condition is not None and stop_condition not in {
            "next_decision",
            "combat_end",
            "room_end",
            "run_end",
        }:
            raise ValueError(
                f"simulation_options.stop_condition {stop_condition!r} is not supported"
            )

    def _validate_decision_payload(self, response: Mapping[str, Any]) -> None:
        self._require_non_empty_str(response, "decision_point_id")
        masked = response.get("masked_emulator_dto")
        if not isinstance(masked, dict):
            raise ApiProtocolError("masked_emulator_dto must be a dictionary")

        for marker in ("terminal", "run_terminal"):
            if marker in masked and not isinstance(masked[marker], bool):
                raise ApiProtocolError(f"masked_emulator_dto.{marker} must be a boolean")

        combat_terminal = masked.get("terminal") is True
        run_terminal = masked.get("run_terminal") is True
        is_terminal = combat_terminal or run_terminal
        if combat_terminal and run_terminal:
            raise ApiProtocolError(
                "masked_emulator_dto.terminal and run_terminal are mutually exclusive"
            )

        if is_terminal:
            outcome = masked.get("outcome")
            valid_outcomes = (
                _RUN_TERMINAL_OUTCOMES if run_terminal else _COMBAT_TERMINAL_OUTCOMES
            )
            if not isinstance(outcome, str) or outcome not in valid_outcomes:
                raise ApiProtocolError(
                    "terminal masked_emulator_dto.outcome must be one of "
                    f"{sorted(valid_outcomes)!r}"
                )
        elif "outcome" in masked:
            raise ApiProtocolError(
                "non-terminal masked_emulator_dto must not include outcome"
            )

        if "legal_actions" not in masked:
            if run_terminal:
                return
            raise ApiProtocolError("masked_emulator_dto.legal_actions must be a list")
        legal_actions = masked["legal_actions"]
        if not isinstance(legal_actions, list):
            raise ApiProtocolError("masked_emulator_dto.legal_actions must be a list")
        if is_terminal and legal_actions:
            raise ApiProtocolError(
                "terminal masked_emulator_dto.legal_actions must be empty"
            )

        action_ids: set[str] = set()
        for index, action in enumerate(legal_actions):
            if not isinstance(action, Mapping):
                raise ApiProtocolError(f"legal_actions[{index}] must be a dictionary")
            action_id = action.get("action_id")
            if not isinstance(action_id, str) or not action_id:
                raise ApiProtocolError(
                    f"legal_actions[{index}].action_id must be a non-empty string"
                )
            if action_id in action_ids:
                raise ApiProtocolError(f"duplicate legal action_id {action_id!r}")
            action_ids.add(action_id)

            if "action_type" in action:
                action_type = action["action_type"]
                if not isinstance(action_type, str) or not action_type:
                    raise ApiProtocolError(
                        f"legal_actions[{index}].action_type must be a non-empty string"
                    )
            if "is_available" in action and not isinstance(action["is_available"], bool):
                raise ApiProtocolError(
                    f"legal_actions[{index}].is_available must be a boolean"
                )
            if "parameters" in action and not isinstance(action["parameters"], Mapping):
                raise ApiProtocolError(
                    f"legal_actions[{index}].parameters must be a dictionary"
                )

    def _normalize_branch_ids(self, branch_ids: Sequence[str]) -> list[str]:
        if isinstance(branch_ids, (str, bytes)):
            raise TypeError("branch_ids must be a sequence of strings")
        normalized = list(branch_ids)
        if not normalized:
            raise ValueError("branch_ids must not be empty")
        for branch_id in normalized:
            self._validate_non_empty_str(branch_id, "branch_id")
            if branch_id == ROOT_BRANCH_ID:
                raise ValueError("root cannot be cancelled, released, or polled")
        if len(set(normalized)) != len(normalized):
            raise ValueError("branch_ids must not contain duplicates")
        return normalized

    @staticmethod
    def _validate_non_empty_str(value: Any, field_name: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field_name} must be a non-empty string")

    @staticmethod
    def _require_non_empty_str(source: Mapping[str, Any], field_name: str) -> str:
        value = source.get(field_name)
        if not isinstance(value, str) or not value:
            raise ApiProtocolError(f"invalid or missing {field_name}")
        return value

    @staticmethod
    def _require_response_match(response: Mapping[str, Any], field_name: str, expected: Any) -> None:
        if response.get(field_name) != expected:
            raise ApiProtocolError(f"response {field_name} does not match request")

    @staticmethod
    def _require_status(response: Mapping[str, Any], accepted: set[str]) -> None:
        if response.get("status") not in accepted:
            raise ApiProtocolError(
                f"unexpected status: {response.get('status')!r}; expected {sorted(accepted)!r}"
            )
