"""Async API v0.5 client for the separately started STS2_RL TCP server."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from sts2_training.api.contract import (
    ApiContract,
    ApiOperationError,
    JsonObject,
    ROOT_BRANCH_ID,
)
from sts2_training.api.tcp_connection import TcpConnection
from sts2_training.selection_log import SelectionEventLogger


class AsyncTrainingApiClient(ApiContract):
    """Async-native API client over a thin ``TcpConnection``.

    DTO construction, validation, correlation, instance tracking, and selection audit
    live in ``ApiContract``. This class owns only async request/response orchestration.
    It intentionally does not implement or depend on the legacy ``RlTransport``.
    """

    def __init__(
        self,
        connection: TcpConnection,
        request_id_factory: Callable[[], str] | None = None,
        *,
        selection_logger: SelectionEventLogger | None = None,
    ) -> None:
        super().__init__(
            request_id_factory=request_id_factory,
            selection_logger=selection_logger,
        )
        self._connection = connection

    async def start_instance(
        self,
        instance_config: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> str:
        request = self._build_start_instance(instance_config)
        response = await self._execute(request, timeout_s=timeout_s)
        return self._accept_start_instance(response)

    async def get_decision(
        self,
        instance_id: str,
        branch_id: str = ROOT_BRANCH_ID,
        *,
        timeout_s: float,
    ) -> JsonObject:
        request = self._build_get_decision(instance_id, branch_id)
        response = await self._execute(request, timeout_s=timeout_s)
        return self._accept_get_decision(response, branch_id)

    async def commit_action(
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
        request = self._build_emulate_action(
            instance_id,
            parent_branch_id,
            branch_id,
            rng_id,
            decision_point_id,
            action_id,
            simulation_options,
        )
        return await self._execute_selected_action(
            request,
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
        request = self._build_close_instance(instance_id)
        response = await self._execute(request, timeout_s=timeout_s)
        return self._accept_close_instance(response)

    async def close(self) -> None:
        await self._connection.close()

    async def __aenter__(self) -> "AsyncTrainingApiClient":
        await self._connection.connect()
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
        request = self._build_branch_batch_operation(
            operation,
            instance_id,
            branch_ids,
        )
        return await self._execute(request, timeout_s=timeout_s)

    async def _execute(
        self,
        request: JsonObject,
        *,
        timeout_s: float,
    ) -> JsonObject:
        response = await self._connection.exchange(
            request,
            timeout_s=timeout_s,
        )
        return self._validate_api_response(request, response)

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
