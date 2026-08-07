from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.selection_log import SelectionAudit, SelectionEventLogger

JsonObject = dict[str, Any]
SCHEMA_VERSION = "0.6"
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
_SIMULATION_OPTION_TYPES = {
    "stop_condition": str,
    "max_depth": int,
    "max_steps": int,
    "max_time_ms": int,
    "max_hypotheses": int,
}
_POSITIVE_INTEGER_SIMULATION_OPTIONS = frozenset(
    {"max_depth", "max_steps", "max_time_ms", "max_hypotheses"}
)
_SUPPORTED_STOP_CONDITIONS = frozenset(
    {"next_decision", "combat_end", "room_end", "run_end"}
)


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
    """DTO v0.6 construction, correlation, active-instance state, and selection audit."""

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

    @property
    def client_session_id(self) -> str:
        return self._client_session_id

    @property
    def instance_id(self) -> str | None:
        return self._instance_id

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
        return self._new_request(request_seq, "start_instance", instance_config=dict(instance_config))

    def _accept_start_instance(self, response: Mapping[str, Any]) -> str:
        self._require_status(response, {"completed"})
        instance_id = self._require_non_empty_str(response, "instance_id")
        self._instance_id = instance_id
        self._audit.clear()
        self._audit.remember(response)
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
        for key, value in options.items():
            expected = _SIMULATION_OPTION_TYPES.get(key)
            if expected is None or value is None:
                continue
            if expected is int:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"simulation_options.{key} must be an integer")
                if key in _POSITIVE_INTEGER_SIMULATION_OPTIONS and value <= 0:
                    raise ValueError(
                        f"simulation_options.{key} must be a positive integer"
                    )
                continue
            if not isinstance(value, expected):
                raise ValueError(
                    f"simulation_options.{key} must be of type {expected.__name__}"
                )
        stop_condition = options.get("stop_condition")
        if stop_condition is not None and stop_condition not in _SUPPORTED_STOP_CONDITIONS:
            raise ValueError(
                f"simulation_options.stop_condition {stop_condition!r} is not supported"
            )

    def _validate_decision_payload(self, response: Mapping[str, Any]) -> None:
        self._require_non_empty_str(response, "decision_point_id")
        if not isinstance(response.get("masked_emulator_dto"), dict):
            raise ApiProtocolError("masked_emulator_dto must be a dictionary")

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
