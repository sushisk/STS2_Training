"""Async-first Training client for RL/Training DTO contract v0.6."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager

from sts2_training.api.contract import (
    ApiContract,
    ApiOperationError,
    ApiProtocolError,
    JsonObject,
    ROOT_BRANCH_ID,
)
from sts2_training.api.tcp_connection import TcpConnection
from sts2_training.api.transport import (
    RetryRequest,
    ServerEpochChangedError,
    TransportError,
)
from sts2_training.selection_log import SelectionEventLogger


class AsyncTrainingApiClient(ApiContract):
    """Serialize one logical request stream over reconnectable asyncio TCP.

    The client advances ``request_seq`` only after observing a definitive API response.
    If a TCP failure happens after send, the exact request becomes ``pending_retry`` and
    every fresh API operation fails closed until that same sequence is replayed.

    RL process restart is not reconciled in-band; a changed ``server_epoch`` permanently
    invalidates this client session and callers must create a new client/session.
    """

    def __init__(
        self,
        connection: TcpConnection,
        *,
        selection_logger: SelectionEventLogger | None = None,
    ) -> None:
        super().__init__(
            client_session_id=connection.client_session_id,
            selection_logger=selection_logger,
        )
        self._connection = connection
        self._operation_lock = asyncio.Lock()
        self._next_request_seq = 1
        self._pending_retry: RetryRequest | None = None
        self._session_invalid = False

    @property
    def next_request_seq(self) -> int:
        return self._next_request_seq

    @property
    def pending_retry(self) -> RetryRequest | None:
        return self._pending_retry

    @property
    def start_uncertain(self) -> bool:
        return self._pending_retry is not None and self._pending_retry.operation == "start_instance"

    @property
    def close_uncertain(self) -> bool:
        return self._pending_retry is not None and self._pending_retry.operation == "close_instance"

    @property
    def session_invalid(self) -> bool:
        return self._session_invalid

    async def retry_request(
        self,
        retry_request: RetryRequest,
        *,
        timeout_s: float,
    ) -> JsonObject | str:
        if self._session_invalid:
            raise RuntimeError("RL server epoch changed; create a new client session")
        if self._pending_retry is None:
            raise RuntimeError("there is no unresolved request to retry")
        if retry_request != self._pending_retry:
            raise ValueError("retry_request does not match the pending request")

        request = retry_request.to_message()
        operation = retry_request.operation
        async with self._operation_deadline(timeout_s) as deadline:
            if operation == "start_instance":
                response = await self._execute(request, deadline=deadline)
                return self._accept_start_instance(response)
            if operation == "get_decision":
                branch_id = request.get("branch_id")
                if not isinstance(branch_id, str) or not branch_id:
                    raise ApiProtocolError("retry get_decision is missing branch_id")
                response = await self._execute(request, deadline=deadline)
                return self._accept_get_decision(response, branch_id)
            if operation == "commit_action":
                return await self._execute_selected_action(
                    request, source_branch_id=ROOT_BRANCH_ID, deadline=deadline
                )
            if operation == "emulate_action":
                parent = request.get("parent_branch_id")
                if not isinstance(parent, str) or not parent:
                    raise ApiProtocolError("retry emulate_action is missing parent_branch_id")
                return await self._execute_selected_action(
                    request, source_branch_id=parent, deadline=deadline
                )
            if operation in {"cancel_branches", "release_branches", "get_branch_status"}:
                return await self._execute(request, deadline=deadline)
            if operation == "close_instance":
                response = await self._execute(request, deadline=deadline)
                return self._accept_close_instance(response)
            raise ValueError(f"unsupported pending operation {operation!r}")

    async def start_instance(
        self,
        instance_config: Mapping[str, object],
        *,
        timeout_s: float,
    ) -> str:
        async with self._operation_deadline(timeout_s) as deadline:
            self._ensure_fresh_request_allowed()
            if self.instance_id is not None:
                raise RuntimeError("client already has an active instance")
            request = self._build_start_instance(self._next_request_seq, instance_config)
            response = await self._execute(request, deadline=deadline)
            return self._accept_start_instance(response)

    async def get_decision(
        self,
        instance_id: str,
        branch_id: str = ROOT_BRANCH_ID,
        *,
        timeout_s: float,
    ) -> JsonObject:
        async with self._operation_deadline(timeout_s) as deadline:
            self._ensure_fresh_request_allowed()
            request = self._build_get_decision(
                self._next_request_seq, instance_id, branch_id
            )
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
            self._ensure_fresh_request_allowed()
            request = self._build_commit_action(
                self._next_request_seq,
                instance_id,
                decision_point_id,
                action_id,
            )
            return await self._execute_selected_action(
                request, source_branch_id=ROOT_BRANCH_ID, deadline=deadline
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
        simulation_options: Mapping[str, object] | None = None,
    ) -> JsonObject:
        async with self._operation_deadline(timeout_s) as deadline:
            self._ensure_fresh_request_allowed()
            request = self._build_emulate_action(
                self._next_request_seq,
                instance_id,
                parent_branch_id,
                branch_id,
                rng_id,
                decision_point_id,
                action_id,
                simulation_options,
            )
            return await self._execute_selected_action(
                request, source_branch_id=parent_branch_id, deadline=deadline
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
        async with self._operation_deadline(timeout_s) as deadline:
            self._ensure_fresh_request_allowed()
            request = self._build_close_instance(self._next_request_seq, instance_id)
            response = await self._execute(request, deadline=deadline)
            return self._accept_close_instance(response)

    async def close(self) -> None:
        async with self._operation_lock:
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
        async with self._operation_deadline(timeout_s) as deadline:
            self._ensure_fresh_request_allowed()
            request = self._build_branch_batch_operation(
                self._next_request_seq, operation, instance_id, branch_ids
            )
            return await self._execute(request, deadline=deadline)

    async def _execute(self, request: JsonObject, *, deadline: float) -> JsonObject:
        token = RetryRequest.from_message(request)
        if token.request_seq != self._next_request_seq:
            raise RuntimeError("request sequence does not match client state")
        if self._pending_retry is not None and token != self._pending_retry:
            raise RuntimeError(
                "an unresolved request is pending; retry the exact request before continuing"
            )
        try:
            response = await self._connection.exchange(request, deadline=deadline)
        except ServerEpochChangedError:
            self._session_invalid = True
            raise
        except asyncio.CancelledError as exc:
            if getattr(exc, "completion_uncertain", False):
                retry = getattr(exc, "retry_request", None)
                self._remember_pending(
                    retry if isinstance(retry, RetryRequest) else token
                )
            raise
        except TransportError as exc:
            if exc.completion_uncertain:
                self._remember_pending(exc.retry_request or token)
            raise

        try:
            validated = self._validate_api_response(request, response)
        except ApiOperationError as exc:
            self._consume_sequence(token)
            if token.operation == "close_instance" and exc.response.get("status") == "faulted":
                self._instance_id = None
                self._audit.clear()
            raise
        except ApiProtocolError:
            await self._connection.invalidate()
            self._remember_pending(token)
            raise

        self._consume_sequence(token)
        return validated

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
        except asyncio.CancelledError as exc:
            self._record_selected_action(
                request,
                source_branch_id=source_branch_id,
                result=response,
                error=exc,
            )
            raise
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
            request, source_branch_id=source_branch_id, result=response
        )
        return response

    def _ensure_fresh_request_allowed(self) -> None:
        if self._session_invalid:
            raise RuntimeError("RL server epoch changed; create a new client session")
        if self._pending_retry is not None:
            raise RuntimeError(
                "an unresolved request is pending; call retry_request() with pending_retry"
            )

    def _remember_pending(self, retry_request: RetryRequest) -> None:
        if retry_request.client_session_id != self.client_session_id:
            raise RuntimeError("pending request belongs to another client session")
        if retry_request.request_seq != self._next_request_seq:
            raise RuntimeError("pending request sequence does not match client state")
        if self._pending_retry is not None and self._pending_retry != retry_request:
            raise RuntimeError("a different unresolved request is already pending")
        self._pending_retry = retry_request

    def _consume_sequence(self, request: RetryRequest) -> None:
        if request.request_seq != self._next_request_seq:
            raise RuntimeError("cannot consume an unexpected request sequence")
        self._pending_retry = None
        self._next_request_seq += 1

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
