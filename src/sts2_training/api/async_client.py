"""Async API v0.5 client for the separately started STS2_RL TCP server."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

from sts2_training.api.contract import (
    ApiContract,
    ApiOperationError,
    ApiProtocolError,
    JsonObject,
    ROOT_BRANCH_ID,
)
from sts2_training.api.tcp_connection import TcpConnection
from sts2_training.api.transport import TransportError
from sts2_training.selection_log import SelectionEventLogger


class AsyncTrainingApiClient(ApiContract):
    """Async-native API client over a thin ``TcpConnection``.

    DTO construction, validation, correlation, instance tracking, and selection audit
    live in ``ApiContract``. This class owns only async request/response orchestration.
    It intentionally does not implement or depend on the legacy ``RlTransport``.

    Public API operations are currently serialized at the client level so the
    single-active-instance and selection-audit state cannot race across tasks.
    Parallel API execution is intentionally deferred until its lifecycle semantics are
    defined explicitly.

    ``timeout_s`` is a total per-operation budget starting when the public API method is
    called. Waiting for the client lock, waiting for the TCP connection lock, connecting,
    writing, and reading the response all consume the same deadline.

    If ``start_instance`` loses a definitive result after the request may have reached
    RL, the client enters a start-uncertain state. Later start attempts are rejected
    instead of risking a second active instance. Failures known to occur before the
    request could reach RL do not enter that state.

    If ``close_instance`` loses a definitive result after send, the client enters a
    close-uncertain state and blocks later API traffic until the caller explicitly
    reconciles the local state with ``reconcile_close_uncertainty``.
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
        self._operation_lock = asyncio.Lock()
        self._start_uncertain = False
        self._close_uncertain = False

    @property
    def start_uncertain(self) -> bool:
        """Whether a previous start may have completed without a known response."""
        return self._start_uncertain

    @property
    def close_uncertain(self) -> bool:
        """Whether a previous close may have completed without a known response."""
        return self._close_uncertain

    def reconcile_close_uncertainty(self, *, assume_closed: bool) -> None:
        """Resolve an ambiguous close using knowledge obtained outside this client.

        ``assume_closed=True`` discards the local active-instance and selection-audit
        state so a later ``start_instance`` can proceed. ``assume_closed=False`` keeps
        the current instance state and permits operations to resume. This method does
        not contact RL; the caller must choose the value only after external
        reconciliation or an explicit operator decision.
        """
        if not self._close_uncertain:
            raise RuntimeError("there is no close_instance uncertainty to reconcile")
        if not isinstance(assume_closed, bool):
            raise TypeError("assume_closed must be a bool")
        if assume_closed:
            self._instance_id = None
            self._audit.clear()
        self._close_uncertain = False

    async def start_instance(
        self,
        instance_config: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> str:
        async with self._operation_deadline(timeout_s) as deadline:
            if self.instance_id is not None:
                raise RuntimeError("client already has an active instance")
            if self._start_uncertain:
                raise RuntimeError(
                    "previous start_instance result is unknown; reconciliation required"
                )
            request = self._build_start_instance(instance_config)
            try:
                response = await self._execute(request, deadline=deadline)
                return self._accept_start_instance(response)
            except asyncio.CancelledError as exc:
                if getattr(exc, "completion_uncertain", False):
                    self._start_uncertain = True
                raise
            except TransportError as exc:
                if exc.completion_uncertain:
                    self._start_uncertain = True
                raise
            except ApiProtocolError:
                # This includes both envelope/correlation failures from _execute() and
                # operation-specific validation failures from _accept_start_instance().
                # In either case a correlated start may already have created an RL
                # instance, so another start must not be sent blindly.
                self._start_uncertain = True
                raise

    async def get_decision(
        self,
        instance_id: str,
        branch_id: str = ROOT_BRANCH_ID,
        *,
        timeout_s: float,
    ) -> JsonObject:
        async with self._operation_deadline(timeout_s) as deadline:
            request = self._build_get_decision(instance_id, branch_id)
            response = await self._execute(request, deadline=deadline)
            return self._accept_get_decision(response, branch_id)

    async def commit_action(
        self,
        instance_id: str,
        decision_point_id: str,
        action_id: str,
        *,
        timeout_s: float,
    ) -> JsonObject:
        async with self._operation_deadline(timeout_s) as deadline:
            request = self._build_commit_action(
                instance_id,
                decision_point_id,
                action_id,
            )
            return await self._execute_selected_action(
                request,
                source_branch_id=ROOT_BRANCH_ID,
                deadline=deadline,
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
        async with self._operation_deadline(timeout_s) as deadline:
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
                deadline=deadline,
            )

    async def cancel_branches(
        self,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> JsonObject:
        async with self._operation_deadline(timeout_s) as deadline:
            return await self._branch_batch_operation(
                "cancel_branches", instance_id, branch_ids, deadline=deadline
            )

    async def release_branches(
        self,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> JsonObject:
        async with self._operation_deadline(timeout_s) as deadline:
            return await self._branch_batch_operation(
                "release_branches", instance_id, branch_ids, deadline=deadline
            )

    async def get_branch_status(
        self,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> JsonObject:
        async with self._operation_deadline(timeout_s) as deadline:
            return await self._branch_batch_operation(
                "get_branch_status", instance_id, branch_ids, deadline=deadline
            )

    async def close_instance(
        self,
        instance_id: str,
        *,
        timeout_s: float,
    ) -> JsonObject:
        async with self._operation_deadline(timeout_s) as deadline:
            request = self._build_close_instance(instance_id)
            try:
                response = await self._execute(request, deadline=deadline)
                return self._accept_close_instance(response)
            except asyncio.CancelledError as exc:
                if getattr(exc, "completion_uncertain", False):
                    self._close_uncertain = True
                raise
            except TransportError as exc:
                if exc.completion_uncertain:
                    self._close_uncertain = True
                raise
            except ApiProtocolError:
                # A correlated-but-invalid close response does not prove whether the
                # instance was closed. Block later traffic until the caller reconciles.
                self._close_uncertain = True
                raise

    async def close(self) -> None:
        async with self._operation_lock:
            await self._connection.close()

    async def __aenter__(self) -> "AsyncTrainingApiClient":
        await self._connection.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    @asynccontextmanager
    async def _operation_deadline(self, timeout_s: float) -> AsyncIterator[float]:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        try:
            async with asyncio.timeout_at(deadline):
                await self._operation_lock.acquire()
        except TimeoutError as exc:
            raise TransportError("RL API call timed out before request started") from exc

        try:
            yield deadline
        finally:
            self._operation_lock.release()

    async def _branch_batch_operation(
        self,
        operation: str,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        deadline: float,
    ) -> JsonObject:
        request = self._build_branch_batch_operation(
            operation,
            instance_id,
            branch_ids,
        )
        return await self._execute(request, deadline=deadline)

    async def _execute(
        self,
        request: JsonObject,
        *,
        deadline: float,
    ) -> JsonObject:
        if self._close_uncertain:
            raise RuntimeError(
                "previous close_instance result is unknown; "
                "reconcile_close_uncertainty() required"
            )
        response = await self._connection.exchange(
            request,
            deadline=deadline,
        )
        try:
            return self._validate_api_response(request, response)
        except ApiProtocolError:
            # A malformed/mismatched response means we can no longer trust that the
            # current stream is aligned with this request. Reconnect before any later
            # API call rather than allowing a stale frame to cascade into more calls.
            await self._connection.invalidate()
            raise

    async def _execute_selected_action(
        self,
        request: JsonObject,
        *,
        source_branch_id: str,
        deadline: float,
    ) -> JsonObject:
        response: JsonObject | None = None
        try:
            response = await self._execute(request, deadline=deadline)
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
