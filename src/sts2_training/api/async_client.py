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
from sts2_training.api.transport import RetryRequest, TransportError
from sts2_training.selection_log import SelectionEventLogger

_STATE_CHANGING_OPERATIONS = frozenset(
    {
        "start_instance",
        "commit_action",
        "emulate_action",
        "cancel_branches",
        "release_branches",
        "close_instance",
    }
)
_UNKNOWN_INSTANCE_FAULT_KIND = "unknown_instance"


class AsyncTrainingApiClient(ApiContract):
    """Async-native API client over a thin ``TcpConnection``.

    DTO construction, validation, correlation, instance tracking, and selection audit
    live in ``ApiContract``. This class owns only async request/response orchestration.
    It intentionally does not implement or depend on the legacy ``RlTransport``.

    Public API operations are currently serialized at the client level so the
    single-active-instance and selection-audit state cannot race across tasks.
    Parallel API execution is intentionally deferred until its lifecycle semantics are
    defined explicitly.

    ``timeout_s`` starts when the public method is called and is the shared deadline for
    waiting on the client lock and the TCP exchange (connection lock, connect, write,
    drain, and response read). Once a response frame has been received, synchronous API
    validation and selection-audit bookkeeping are not treated as transport timeout
    phases; ``timeout_s`` is therefore not a hard wall-clock cap on post-response work.

    If a state-changing request loses a definitive result after it may have reached RL,
    the client stores the exact serialized request as ``pending_retry`` and fails closed:
    unrelated API requests are blocked until that same request is replayed with
    ``retry_request()`` or an operation-specific reconciliation method explicitly clears
    the uncertainty.
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
        self._pending_retry: RetryRequest | None = None

    @property
    def start_uncertain(self) -> bool:
        """Whether a previous start may have completed without a known response."""
        return self._start_uncertain

    @property
    def close_uncertain(self) -> bool:
        """Whether a previous close may have completed without a known response."""
        return self._close_uncertain

    @property
    def pending_retry(self) -> RetryRequest | None:
        """Exact state-changing request that must be replayed before new API traffic."""
        return self._pending_retry

    def reconcile_start_uncertainty(self, *, instance_id: str | None) -> None:
        """Resolve an ambiguous start using knowledge obtained outside this client.

        Pass the externally confirmed active ``instance_id`` to adopt it locally, or
        ``None`` only when external reconciliation establishes that no instance was
        created. This method does not contact RL.
        """
        if not self._start_uncertain:
            raise RuntimeError("there is no start_instance uncertainty to reconcile")
        if instance_id is not None:
            self._validate_non_empty_str(instance_id, "instance_id")
        self._instance_id = instance_id
        self._audit.clear()
        self._start_uncertain = False
        self._clear_pending_operation("start_instance")

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
        self._clear_pending_operation("close_instance")

    async def retry_request(
        self,
        retry_request: RetryRequest,
        *,
        timeout_s: float,
    ) -> JsonObject | str:
        """Replay the current completion-uncertain request with the same request id.

        Only the exact token exposed by ``pending_retry`` is accepted. This prevents a
        caller from accidentally creating a fresh logical request while RL may already
        have applied the original state change.
        """
        if not isinstance(retry_request, RetryRequest):
            raise TypeError("retry_request must be a RetryRequest")
        if self._pending_retry is None:
            raise RuntimeError("there is no completion-uncertain request to retry")
        if retry_request != self._pending_retry:
            raise ValueError("retry_request does not match the pending request")

        request = retry_request.to_message()
        operation = request.get("operation")
        async with self._operation_deadline(timeout_s) as deadline:
            if operation == "start_instance":
                try:
                    response = await self._execute(request, deadline=deadline)
                    return self._accept_start_instance(response)
                except ApiProtocolError:
                    self._remember_uncertain(request, retry_request)
                    raise

            if operation == "close_instance":
                try:
                    response = await self._execute(request, deadline=deadline)
                    return self._accept_close_instance(response)
                except ApiProtocolError:
                    self._remember_uncertain(request, retry_request)
                    raise

            if operation == "commit_action":
                return await self._execute_selected_action(
                    request,
                    source_branch_id=ROOT_BRANCH_ID,
                    deadline=deadline,
                )

            if operation == "emulate_action":
                source_branch_id = request.get("parent_branch_id")
                if not isinstance(source_branch_id, str) or not source_branch_id:
                    raise ApiProtocolError(
                        "retry emulate_action request is missing parent_branch_id"
                    )
                return await self._execute_selected_action(
                    request,
                    source_branch_id=source_branch_id,
                    deadline=deadline,
                )

            if operation in {"cancel_branches", "release_branches"}:
                return await self._execute(request, deadline=deadline)

            raise ValueError(
                f"pending retry operation {operation!r} is not state-changing"
            )

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
                    "previous start_instance result is unknown; "
                    "retry_request() or reconciliation required"
                )
            request = self._build_start_instance(instance_config)
            try:
                response = await self._execute(request, deadline=deadline)
                return self._accept_start_instance(response)
            except ApiProtocolError:
                # This includes both envelope/correlation failures from _execute() and
                # operation-specific validation failures from _accept_start_instance().
                # In either case a correlated start may already have created an RL
                # instance, so another start must not be sent blindly.
                self._remember_uncertain(request)
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
            except ApiProtocolError:
                # A correlated-but-invalid close response does not prove whether the
                # instance was closed. Block later traffic until the caller reconciles
                # or replays the exact same request.
                self._remember_uncertain(request)
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
        request_token = RetryRequest.from_message(request)
        if self._pending_retry is not None and request_token != self._pending_retry:
            raise RuntimeError(
                "a completion-uncertain state-changing request is pending; "
                "retry_request() or explicit reconciliation is required"
            )

        try:
            response = await self._connection.exchange(
                request,
                deadline=deadline,
            )
        except asyncio.CancelledError as exc:
            if getattr(exc, "completion_uncertain", False):
                token = getattr(exc, "retry_request", None)
                self._remember_uncertain(
                    request,
                    token if isinstance(token, RetryRequest) else None,
                )
            raise
        except TransportError as exc:
            if exc.completion_uncertain:
                self._remember_uncertain(request, exc.retry_request)
            raise

        try:
            validated = self._validate_api_response(request, response)
        except ApiOperationError as exc:
            if self._is_evicted_close_replay_rejection(request_token, exc):
                # A bounded RL close tombstone can be evicted before a lost close
                # response is replayed. In that case unknown_instance does not prove
                # whether the original close succeeded, so keep the exact replay token
                # and close uncertainty until explicit reconciliation.
                self._remember_uncertain(request, request_token)
            else:
                # Other rejected/faulted responses are definitive for this logical
                # request and reconcile any prior transport uncertainty.
                self._clear_pending_if_matches(request_token)
            raise
        except ApiProtocolError:
            # A malformed/mismatched response means we can no longer trust that the
            # current stream is aligned with this request. Reconnect before any later
            # API call rather than allowing a stale frame to cascade into more calls.
            await self._connection.invalidate()
            self._remember_uncertain(request, request_token)
            raise

        self._clear_pending_if_matches(request_token)
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
            # CancelledError is a BaseException in modern Python and would bypass the
            # generic Exception branch. Record the selection explicitly because RL may
            # already have applied the action when cancellation reaches the client.
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
        except ApiProtocolError as exc:
            self._remember_uncertain(request)
            self._record_selected_action(
                request,
                source_branch_id=source_branch_id,
                result=response,
                error=exc,
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

    def _remember_uncertain(
        self,
        request: Mapping[str, Any],
        retry_request: RetryRequest | None = None,
    ) -> None:
        operation = request.get("operation")
        if operation not in _STATE_CHANGING_OPERATIONS:
            return

        token = retry_request or RetryRequest.from_message(request)
        if self._pending_retry is not None and self._pending_retry != token:
            # Public calls are serialized and new requests are blocked while a token is
            # pending, so this should be unreachable. Preserve the older uncertainty
            # rather than overwriting the only safe replay handle.
            return
        self._pending_retry = token
        if operation == "start_instance":
            self._start_uncertain = True
        elif operation == "close_instance":
            self._close_uncertain = True

    def _is_evicted_close_replay_rejection(
        self,
        retry_request: RetryRequest,
        exc: ApiOperationError,
    ) -> bool:
        return (
            self._pending_retry == retry_request
            and retry_request.operation == "close_instance"
            and exc.response.get("status") == "rejected"
            and exc.response.get("fault_kind") == _UNKNOWN_INSTANCE_FAULT_KIND
        )

    def _clear_pending_if_matches(self, retry_request: RetryRequest) -> None:
        if self._pending_retry == retry_request:
            operation = retry_request.operation
            self._pending_retry = None
            if operation == "start_instance":
                self._start_uncertain = False
            elif operation == "close_instance":
                self._close_uncertain = False

    def _clear_pending_operation(self, operation: str) -> None:
        if self._pending_retry is not None and self._pending_retry.operation == operation:
            self._pending_retry = None
