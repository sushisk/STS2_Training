from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from sts2_training.api.contract import (
    ApiContract,
    ApiOperationError,
    ApiProtocolError,
    JsonObject,
    RequestFaultedError,
    RequestRejectedError,
    ROOT_BRANCH_ID,
    ROOT_RNG_ID,
    SCHEMA_VERSION,
)
from sts2_training.api.transport import RlTransport
from sts2_training.selection_log import SelectionEventLogger


class TrainingApiClient(ApiContract):
    """Synchronous API v0.5 client for the legacy local-process transport."""

    def __init__(
        self,
        transport: RlTransport,
        request_id_factory: Callable[[], str] | None = None,
        *,
        selection_logger: SelectionEventLogger | None = None,
    ) -> None:
        super().__init__(
            request_id_factory=request_id_factory,
            selection_logger=selection_logger,
        )
        self._transport = transport

    def start_instance(
        self,
        instance_config: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> str:
        request = self._build_start_instance(instance_config)
        response = self._execute(request, timeout_s=timeout_s)
        return self._accept_start_instance(response)

    def get_decision(
        self,
        instance_id: str,
        branch_id: str = ROOT_BRANCH_ID,
        *,
        timeout_s: float,
    ) -> JsonObject:
        request = self._build_get_decision(instance_id, branch_id)
        response = self._execute(request, timeout_s=timeout_s)
        return self._accept_get_decision(response, branch_id)

    def commit_action(
        self,
        instance_id: str,
        decision_point_id: str,
        action_id: str,
        *,
        timeout_s: float,
    ) -> JsonObject:
        request = self._build_commit_action(
            instance_id,
            decision_point_id,
            action_id,
        )
        return self._execute_selected_action(
            request,
            source_branch_id=ROOT_BRANCH_ID,
            timeout_s=timeout_s,
        )

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
        request = self._build_emulate_action(
            instance_id,
            parent_branch_id,
            branch_id,
            rng_id,
            decision_point_id,
            action_id,
            simulation_options,
        )
        return self._execute_selected_action(
            request,
            source_branch_id=parent_branch_id,
            timeout_s=timeout_s,
        )

    def cancel_branches(
        self,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> JsonObject:
        return self._branch_batch_operation(
            "cancel_branches", instance_id, branch_ids, timeout_s=timeout_s
        )

    def release_branches(
        self,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> JsonObject:
        return self._branch_batch_operation(
            "release_branches", instance_id, branch_ids, timeout_s=timeout_s
        )

    def get_branch_status(
        self,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> JsonObject:
        return self._branch_batch_operation(
            "get_branch_status", instance_id, branch_ids, timeout_s=timeout_s
        )

    def close_instance(
        self,
        instance_id: str,
        *,
        timeout_s: float,
    ) -> JsonObject:
        request = self._build_close_instance(instance_id)
        response = self._execute(request, timeout_s=timeout_s)
        return self._accept_close_instance(response)

    def close(self) -> None:
        self._transport.close()

    def _branch_batch_operation(
        self,
        operation: str,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> JsonObject:
        request = self._build_branch_batch_operation(
            operation,
            instance_id,
            branch_ids,
        )
        return self._execute(request, timeout_s=timeout_s)

    def _execute(self, request: JsonObject, *, timeout_s: float) -> JsonObject:
        response = self._transport.call(request, timeout_s=timeout_s)
        return self._validate_api_response(request, response)

    def _execute_selected_action(
        self,
        request: JsonObject,
        *,
        source_branch_id: str,
        timeout_s: float,
    ) -> JsonObject:
        response: JsonObject | None = None
        try:
            response = self._execute(request, timeout_s=timeout_s)
            self._validate_selected_action_response(request, response)
        except ApiOperationError as exc:
            self._record_selected_action(
                request,
                source_branch_id=source_branch_id,
                result=exc.response,
            )
            raise
        except Exception as exc:
            self._record_selected_action(
                request,
                source_branch_id=source_branch_id,
                result=response,
                error=exc,
            )
            raise

        self._record_selected_action(
            request,
            source_branch_id=source_branch_id,
            result=response,
        )
        return response


__all__ = [
    "ApiOperationError",
    "ApiProtocolError",
    "RequestFaultedError",
    "RequestRejectedError",
    "ROOT_BRANCH_ID",
    "ROOT_RNG_ID",
    "SCHEMA_VERSION",
    "TrainingApiClient",
]
