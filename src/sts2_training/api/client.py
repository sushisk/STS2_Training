from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from sts2_training.api.transport import JsonObject, RlTransport

SCHEMA_VERSION = "0.5"

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


class ApiProtocolError(RuntimeError):
    """The RL response violated the communication contract or correlation rules."""


class TrainingApiClient:
    """Builds RL–Training API v0.5 requests and validates their responses.

    This stage intentionally keeps DTOs as JSON-safe dictionaries. Pydantic models,
    retry, and the real local-process transport are introduced later.
    """

    def __init__(
        self,
        transport: RlTransport,
        request_id_factory: Callable[[], str],
    ) -> None:
        self._transport = transport
        self._request_id_factory = request_id_factory

    def start_instance(
        self,
        instance_config: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> str:
        request = self._new_request(
            "start_instance",
            instance_config=dict(instance_config),
        )
        response = self._execute(request, timeout_s=timeout_s)
        self._require_status(response, {"completed"})

        instance_id = self._require_non_empty_str(response, "instance_id")
        return instance_id

    def get_decision(
        self,
        instance_id: str,
        branch_id: str,
        *,
        timeout_s: float,
    ) -> JsonObject:
        self._validate_non_empty_str(instance_id, "instance_id")
        self._validate_non_empty_str(branch_id, "branch_id")

        request = self._new_request(
            "get_decision",
            instance_id=instance_id,
            branch_id=branch_id,
        )
        response = self._execute(request, timeout_s=timeout_s)

        # A non-completed response is still a valid protocol response. The
        # application layer will decide how to handle rejected/faulted/etc.
        if response["status"] == "completed":
            self._require_response_match(response, "branch_id", branch_id)
            self._validate_decision_payload(response)

        return response

    def commit_action(
        self,
        instance_id: str,
        decision_point_id: str,
        action_id: str,
        *,
        timeout_s: float,
    ) -> JsonObject:
        self._validate_non_empty_str(instance_id, "instance_id")
        self._validate_non_empty_str(decision_point_id, "decision_point_id")
        self._validate_non_empty_str(action_id, "action_id")

        request = self._new_request(
            "commit_action",
            instance_id=instance_id,
            branch_id="root",
            rng_id=0,
            decision_point_id=decision_point_id,
            action_id=action_id,
        )
        response = self._execute(request, timeout_s=timeout_s)

        if response["status"] == "completed":
            # A successful root commit reaches the next decision or terminal state.
            if "branch_id" in response:
                self._require_response_match(response, "branch_id", "root")
            self._validate_decision_payload(response)

        return response

    def emulate_action(
        self,
        instance_id: str,
        parent_branch_id: str,
        branch_id: str,
        rng_id: int,
        decision_point_id: str,
        action_id: str,
        *,
        timeout_s: float,
        simulation_options: Mapping[str, Any] | None = None,
    ) -> JsonObject:
        self._validate_non_empty_str(instance_id, "instance_id")
        self._validate_non_empty_str(parent_branch_id, "parent_branch_id")
        self._validate_non_empty_str(branch_id, "branch_id")
        self._validate_non_empty_str(decision_point_id, "decision_point_id")
        self._validate_non_empty_str(action_id, "action_id")

        if branch_id == "root":
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

        request = self._new_request("emulate_action", **fields)
        response = self._execute(request, timeout_s=timeout_s)

        # queued/running may not yet contain a Decision payload.
        if response["status"] in {"completed", "partial", "queued", "running"}:
            self._require_response_match(response, "branch_id", branch_id)
            if "parent_branch_id" in response:
                self._require_response_match(
                    response,
                    "parent_branch_id",
                    parent_branch_id,
                )
            if "rng_id" in response and response["rng_id"] != rng_id:
                raise ApiProtocolError("response rng_id does not match request")

        if response["status"] in {"completed", "partial"}:
            self._validate_decision_payload(response)

        return response

    def cancel_branches(
        self,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> JsonObject:
        return self._branch_batch_operation(
            "cancel_branches",
            instance_id,
            branch_ids,
            timeout_s=timeout_s,
        )

    def release_branches(
        self,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> JsonObject:
        return self._branch_batch_operation(
            "release_branches",
            instance_id,
            branch_ids,
            timeout_s=timeout_s,
        )

    def get_branch_status(
        self,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> JsonObject:
        return self._branch_batch_operation(
            "get_branch_status",
            instance_id,
            branch_ids,
            timeout_s=timeout_s,
        )

    def close_instance(
        self,
        instance_id: str,
        *,
        timeout_s: float,
    ) -> JsonObject:
        self._validate_non_empty_str(instance_id, "instance_id")
        request = self._new_request("close_instance", instance_id=instance_id)
        return self._execute(request, timeout_s=timeout_s)

    def close(self) -> None:
        """Close the communication transport, not an RL instance."""

        self._transport.close()

    def _branch_batch_operation(
        self,
        operation: str,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> JsonObject:
        self._validate_non_empty_str(instance_id, "instance_id")
        normalized_ids = self._normalize_branch_ids(branch_ids)

        request = self._new_request(
            operation,
            instance_id=instance_id,
            branch_ids=normalized_ids,
        )
        return self._execute(request, timeout_s=timeout_s)

    def _new_request(self, operation: str, **fields: Any) -> JsonObject:
        request_id = self._request_id_factory()
        self._validate_non_empty_str(request_id, "request_id")
        return {
            "schema_version": SCHEMA_VERSION,
            "request_id": request_id,
            "operation": operation,
            **fields,
        }

    def _execute(self, request: JsonObject, *, timeout_s: float) -> JsonObject:
        response = self._transport.call(request, timeout_s=timeout_s)
        if not isinstance(response, dict):
            raise ApiProtocolError("response must be a dictionary")

        self._validate_envelope(request, response)
        return response

    def _validate_envelope(
        self,
        request: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> None:
        if response.get("schema_version") != SCHEMA_VERSION:
            raise ApiProtocolError("schema_version does not match")
        if response.get("request_id") != request.get("request_id"):
            raise ApiProtocolError("request_id does not match")
        if response.get("operation") != request.get("operation"):
            raise ApiProtocolError("operation does not match")

        status = self._require_non_empty_str(response, "status")
        if status not in KNOWN_STATUSES:
            raise ApiProtocolError(f"unknown status: {status}")

        request_instance_id = request.get("instance_id")
        if request_instance_id is not None and "instance_id" in response:
            if response.get("instance_id") != request_instance_id:
                raise ApiProtocolError("instance_id does not match")

        if status in {"rejected", "faulted"}:
            self._require_non_empty_str(response, "error")
        if status == "faulted":
            self._require_non_empty_str(response, "fault_kind")

    def _validate_decision_payload(self, response: Mapping[str, Any]) -> None:
        self._require_non_empty_str(response, "decision_point_id")
        masked_emulator_dto = response.get("masked_emulator_dto")
        if not isinstance(masked_emulator_dto, dict):
            raise ApiProtocolError("masked_emulator_dto must be a dictionary")

    def _validate_simulation_options(
        self,
        options: Mapping[str, Any],
    ) -> None:
        if not isinstance(options, Mapping):
            raise TypeError("simulation_options must be a mapping")

        positive_integer_fields = {
            "max_depth",
            "max_steps",
            "max_time_ms",
            "max_hypotheses",
        }
        for field in positive_integer_fields:
            if field not in options:
                continue
            value = options[field]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")

        if "stop_condition" in options:
            self._validate_non_empty_str(options["stop_condition"], "stop_condition")

    def _normalize_branch_ids(self, branch_ids: Sequence[str]) -> list[str]:
        if isinstance(branch_ids, (str, bytes)):
            raise TypeError("branch_ids must be a sequence of strings")

        normalized = list(branch_ids)
        if not normalized:
            raise ValueError("branch_ids must not be empty")

        for branch_id in normalized:
            self._validate_non_empty_str(branch_id, "branch_id")
            if branch_id == "root":
                raise ValueError("root cannot be cancelled, released, or polled")

        if len(set(normalized)) != len(normalized):
            raise ValueError("branch_ids must not contain duplicates")

        return normalized

    @staticmethod
    def _validate_non_empty_str(value: Any, field_name: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field_name} must be a non-empty string")

    @staticmethod
    def _require_non_empty_str(
        source: Mapping[str, Any],
        field_name: str,
    ) -> str:
        value = source.get(field_name)
        if not isinstance(value, str) or not value:
            raise ApiProtocolError(f"invalid or missing {field_name}")
        return value

    @staticmethod
    def _require_response_match(
        response: Mapping[str, Any],
        field_name: str,
        expected: Any,
    ) -> None:
        if response.get(field_name) != expected:
            raise ApiProtocolError(f"response {field_name} does not match request")

    @staticmethod
    def _require_status(
        response: Mapping[str, Any],
        accepted: set[str],
    ) -> None:
        if response.get("status") not in accepted:
            raise ApiProtocolError(
                f"unexpected status: {response.get('status')!r}; "
                f"expected one of {sorted(accepted)!r}"
            )
