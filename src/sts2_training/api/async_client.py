"""Async API v0.5 client for a separately started STS2_RL TCP server."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from sts2_training.api.client import (
    ApiOperationError,
    ApiProtocolError,
    RequestFaultedError,
    RequestRejectedError,
    ROOT_BRANCH_ID,
    ROOT_RNG_ID,
    TrainingApiClient,
)
from sts2_training.api.transport import JsonObject
from sts2_training.selection_log import SelectionEventLogger


class AsyncRlTransport(Protocol):
    async def call(
        self,
        request: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> JsonObject: ...

    def is_alive(self) -> bool: ...

    async def close(self) -> None: ...


class AsyncTrainingApiClient(TrainingApiClient):
    """Async counterpart of ``TrainingApiClient`` using the same API v0.5 DTO rules.

    Request construction, DTO validation, response correlation, and selection-audit
    behavior are inherited from the synchronous client. Only transport execution is
    asynchronous, so the two clients keep one contract implementation.
    """

    def __init__(
        self,
        transport: AsyncRlTransport,
        request_id_factory: Callable[[], str] | None = None,
        *,
        selection_logger: SelectionEventLogger | None = None,
    ) -> None:
        super().__init__(
            transport,  # type: ignore[arg-type]
            request_id_factory=request_id_factory,
            selection_logger=selection_logger,
        )
        self._transport = transport

    async def start_instance(
        self,
        instance_config: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> str:
        request = self._new_request(
            "start_instance",
            instance_config=dict(instance_config),
        )
        response = await self._execute(request, timeout_s=timeout_s)
        self._require_status(response, {"completed"})
        instance_id = self._require_non_empty_str(response, "instance_id")
        self._instance_id = instance_id
        self._audit.clear()
        self._audit.remember(response)
        return instance_id

    async def get_decision(
        self,
        instance_id: str,
        branch_id: str = ROOT_BRANCH_ID,
        *,
        timeout_s: float,
    ) -> JsonObject:
        self._validate_instance_id(instance_id)
        self._validate_non_empty_str(branch_id, "branch_id")
        request = self._new_request(
            "get_decision",
            instance_id=instance_id,
            branch_id=branch_id,
        )
        response = await self._execute(request, timeout_s=timeout_s)
        if response["status"] == "completed":
            self._require_response_match(response, "branch_id", branch_id)
            self._validate_decision_payload(response)
            self._audit.remember(response)
        return response

    async def commit_action(
        self,
        instance_id: str,
        decision_point_id: str,
        action_id: str,
        *,
        timeout_s: float,
    ) -> JsonObject:
        self._validate_instance_id(instance_id)
        self._validate_non_empty_str(decision_point_id, "decision_point_id")
        self._validate_non_empty_str(action_id, "action_id")
        request = self._new_request(
            "commit_action",
            instance_id=instance_id,
            branch_id=ROOT_BRANCH_ID,
            rng_id=ROOT_RNG_ID,
            decision_point_id=decision_point_id,
            action_id=action_id,
        )
        return await self._execute_selected_action(
            request,
            source_branch_id=ROOT_BRANCH_ID,
            timeout_s=timeout_s,
        )

    async def emulate_action(
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
        self._validate_instance_id(instance_id)
        self._validate_non_empty_str(parent_branch_id, "parent_branch_id")
        self._validate_non_empty_str(branch_id, "branch_id")
        self._validate_non_empty_str(decision_point_id, "decision_point_id")
        self._validate_non_empty_str(action_id, "action_id")
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

        return await self._execute_selected_action(
            self._new_request("emulate_action", **fields),
            source_branch_id=parent_branch_id,
            timeout_s=timeout_s,
        )

    async def cancel_branches(
        self,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> JsonObject:
        return await self._branch_batch_operation(
            "cancel_branches", instance_id, branch_ids, timeout_s=timeout_s
        )

    async def release_branches(
        self,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> JsonObject:
        return await self._branch_batch_operation(
            "release_branches", instance_id, branch_ids, timeout_s=timeout_s
        )

    async def get_branch_status(
        self,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> JsonObject:
        return await self._branch_batch_operation(
            "get_branch_status", instance_id, branch_ids, timeout_s=timeout_s
        )

    async def close_instance(
        self,
        instance_id: str,
        *,
        timeout_s: float,
    ) -> JsonObject:
        self._validate_instance_id(instance_id)
        request = self._new_request("close_instance", instance_id=instance_id)
        response = await self._execute(request, timeout_s=timeout_s)
        self._require_status(response, {"completed"})
        self._instance_id = None
        self._audit.clear()
        return response

    async def close(self) -> None:
        await self._transport.close()

    async def __aenter__(self) -> "AsyncTrainingApiClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _branch_batch_operation(
        self,
        operation: str,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> JsonObject:
        self._validate_instance_id(instance_id)
        normalized_ids = self._normalize_branch_ids(branch_ids)
        request = self._new_request(
            operation,
            instance_id=instance_id,
            branch_ids=normalized_ids,
        )
        return await self._execute(request, timeout_s=timeout_s)

    async def _execute(
        self,
        request: JsonObject,
        *,
        timeout_s: float,
    ) -> JsonObject:
        response = await self._transport.call(request, timeout_s=timeout_s)
        if not isinstance(response, dict):
            raise ApiProtocolError("response must be a dictionary")
        self._validate_envelope(request, response)
        if response["status"] == "rejected":
            raise RequestRejectedError(response)
        if response["status"] == "faulted":
            raise RequestFaultedError(response)
        return response

    async def _execute_selected_action(
        self,
        request: JsonObject,
        *,
        source_branch_id: str,
        timeout_s: float,
    ) -> JsonObject:
        response: JsonObject | None = None
        try:
            response = await self._execute(request, timeout_s=timeout_s)
            self._validate_selected_action_response(request, response)
        except ApiOperationError as exc:
            self._audit.record_action(
                request, source_branch_id=source_branch_id, result=exc.response
            )
            raise
        except Exception as exc:
            self._audit.record_action(
                request, source_branch_id=source_branch_id, result=response, error=exc
            )
            raise

        self._audit.record_action(
            request, source_branch_id=source_branch_id, result=response
        )
        return response
