"""Async-first Training client for RL/Training DTO contract v0.7."""

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

_NON_CONSUMING_SESSION_REJECTIONS = frozenset(
    {
        "invalid_request",
        "session_sequence_conflict",
        "session_sequence_gap",
        "session_capacity_exceeded",
    }
)
_FATAL_CONSUMING_SESSION_REJECTIONS = frozenset(
    {
        "session_instance_conflict",
        "unknown_instance",
    }
)


class AsyncTrainingApiClient(ApiContract):
    """Serialize one logical request stream over reconnectable asyncio TCP.

    ``request_seq`` advances only after the complete operation-specific DTO response has
    been validated. If transport or protocol validation becomes completion-uncertain,
    the exact current request is retained as ``pending_retry`` and fresh requests fail
    closed.

    A changed RL ``server_epoch`` or a session-level sequencing/ownership rejection
    permanently invalidates this logical client. v0.7 intentionally does not guess how
    to continue from divergent client/server session state.
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
        self._emulate_actions_boundaries: frozenset[str] = frozenset()
        self._pending_retry_cleanup: (
            tuple[RetryRequest, str, tuple[str, ...]] | None
        ) = None

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

    @property
    def emulate_actions_boundaries(self) -> frozenset[str]:
        """Semantic boundaries the active RL instance accepts via ``emulate_actions``.

        Older v0.7 servers omit this optional capability; an empty set therefore means
        callers must not infer semantic batch support merely from the numeric batch size.
        """

        return self._emulate_actions_boundaries

    def defer_branch_cleanup_after_retry(
        self,
        retry_request: RetryRequest,
        instance_id: str,
        branch_ids: Sequence[str],
    ) -> None:
        """Attach Branch ownership cleanup to an unresolved exact-replay request.

        ``branch_ids`` may contain both Branches that were definitely created before the
        uncertain request and Branches that the uncertain request itself may have
        created. The latter set is inferred from the exact serialized retry request, so a
        definitive replay ``rejected`` outcome can release only the already-owned IDs;
        replay ``completed``/``faulted`` outcomes release both sets.

        A caller must not send ``release_branches`` while the source request is
        completion-uncertain. Once exact replay resolves, cleanup is sent as the
        immediately following sequence. If cleanup itself becomes uncertain, its exact
        request becomes ``pending_retry`` and remains recoverable.
        """

        if self._session_invalid:
            raise RuntimeError("RL session is invalid; cannot defer Branch cleanup")
        if self._pending_retry is None or retry_request != self._pending_retry:
            raise ValueError("cleanup must be attached to the current pending_retry")

        request = retry_request.to_message()
        if request.get("instance_id") != instance_id:
            raise ValueError("deferred cleanup instance_id does not match pending request")
        if retry_request.operation not in {"emulate_action", "emulate_actions"}:
            raise ValueError("deferred Branch cleanup requires an emulate request")

        normalized: list[str] = []
        seen: set[str] = set()
        for branch_id in branch_ids:
            if not isinstance(branch_id, str) or not branch_id or branch_id == ROOT_BRANCH_ID:
                raise ValueError("deferred cleanup branch_ids must be non-root strings")
            if branch_id not in seen:
                seen.add(branch_id)
                normalized.append(branch_id)
        if not normalized:
            raise ValueError("deferred cleanup branch_ids must not be empty")

        expected = self._branch_ids_created_by_retry(retry_request)
        if not expected or not expected <= seen:
            raise ValueError(
                "deferred cleanup must include every Branch that the pending emulate "
                "request may have created"
            )

        record = (retry_request, instance_id, tuple(normalized))
        if self._pending_retry_cleanup is not None and self._pending_retry_cleanup != record:
            raise RuntimeError("a different deferred Branch cleanup is already pending")
        self._pending_retry_cleanup = record

    def _accept_start_instance_for_request(
        self,
        request: Mapping[str, object],
        response: Mapping[str, object],
    ) -> str:
        instance_config = request.get("instance_config")
        if (
            isinstance(instance_config, Mapping)
            and instance_config.get("instance_type") == "combat"
            and response.get("max_emulate_actions_items") is None
        ):
            raise ApiProtocolError(
                "combat start_instance response must include max_emulate_actions_items"
            )

        raw_boundaries = response.get("emulate_actions_boundaries", [])
        if raw_boundaries is None:
            raw_boundaries = []
        if not isinstance(raw_boundaries, list) or any(
            not isinstance(boundary, str) or not boundary
            for boundary in raw_boundaries
        ):
            raise ApiProtocolError(
                "emulate_actions_boundaries must be an array of non-empty strings"
            )
        if len(set(raw_boundaries)) != len(raw_boundaries):
            raise ApiProtocolError("emulate_actions_boundaries must not contain duplicates")

        instance_id = self._accept_start_instance(response)
        self._emulate_actions_boundaries = frozenset(raw_boundaries)
        self._pending_retry_cleanup = None
        return instance_id

    async def retry_request(
        self,
        retry_request: RetryRequest,
        *,
        timeout_s: float,
    ) -> JsonObject | str:
        if self._session_invalid:
            raise RuntimeError("RL session is invalid; create a new client session")
        if self._pending_retry is None:
            raise RuntimeError("there is no unresolved request to retry")
        if retry_request != self._pending_retry:
            raise ValueError("retry_request does not match the pending request")

        request = retry_request.to_message()
        operation = retry_request.operation
        async with self._operation_deadline(timeout_s) as deadline:
            if operation == "start_instance":
                response = await self._execute(request, deadline=deadline)
                try:
                    result = self._accept_start_instance_for_request(request, response)
                except ApiProtocolError:
                    await self._mark_protocol_uncertain(request)
                    raise
                self._consume_sequence(retry_request)
                return result

            if operation == "get_decision":
                branch_id = request.get("branch_id")
                if not isinstance(branch_id, str) or not branch_id:
                    raise ApiProtocolError("retry get_decision is missing branch_id")
                response = await self._execute(request, deadline=deadline)
                try:
                    result = self._accept_get_decision(response, branch_id)
                except ApiProtocolError:
                    await self._mark_protocol_uncertain(request)
                    raise
                self._consume_sequence(retry_request)
                return result

            if operation == "commit_action":
                return await self._execute_selected_action(
                    request, source_branch_id=ROOT_BRANCH_ID, deadline=deadline
                )

            if operation == "emulate_action":
                parent = request.get("parent_branch_id")
                if not isinstance(parent, str) or not parent:
                    raise ApiProtocolError("retry emulate_action is missing parent_branch_id")
                try:
                    result = await self._execute_selected_action(
                        request, source_branch_id=parent, deadline=deadline
                    )
                except BaseException as primary_error:
                    if self._pending_retry is None and not self._session_invalid:
                        replay_rejected = (
                            isinstance(primary_error, ApiOperationError)
                            and primary_error.response.get("status") == "rejected"
                        )
                        try:
                            await self._complete_deferred_branch_cleanup(
                                retry_request,
                                deadline=deadline,
                                include_retry_created=not replay_rejected,
                            )
                        except Exception as cleanup_error:
                            if replay_rejected:
                                raise primary_error from cleanup_error
                            raise cleanup_error from primary_error
                    raise
                await self._complete_deferred_branch_cleanup(
                    retry_request,
                    deadline=deadline,
                )
                return result

            if operation == "emulate_actions":
                items = request.get("items")
                if not isinstance(items, list) or not items:
                    raise ApiProtocolError("retry emulate_actions is missing items")
                try:
                    result = await self._execute_emulate_actions(
                        request, deadline=deadline
                    )
                except BaseException as primary_error:
                    if self._pending_retry is None and not self._session_invalid:
                        replay_rejected = (
                            isinstance(primary_error, ApiOperationError)
                            and primary_error.response.get("status") == "rejected"
                        )
                        try:
                            await self._complete_deferred_branch_cleanup(
                                retry_request,
                                deadline=deadline,
                                include_retry_created=not replay_rejected,
                            )
                        except Exception as cleanup_error:
                            if replay_rejected:
                                raise primary_error from cleanup_error
                            raise cleanup_error from primary_error
                    raise
                await self._complete_deferred_branch_cleanup(
                    retry_request,
                    deadline=deadline,
                )
                return result

            if operation in {"cancel_branches", "release_branches", "get_branch_status"}:
                response = await self._execute(request, deadline=deadline)
                self._consume_sequence(retry_request)
                if (
                    self._pending_retry_cleanup is not None
                    and self._pending_retry_cleanup[0] == retry_request
                ):
                    self._pending_retry_cleanup = None
                return response

            if operation == "close_instance":
                response = await self._execute(request, deadline=deadline)
                try:
                    result = self._accept_close_instance(response)
                except ApiProtocolError:
                    await self._mark_protocol_uncertain(request)
                    raise
                self._consume_sequence(retry_request)
                self._clear_instance_extensions()
                return result

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
            if (
                instance_config.get("instance_type") == "whole_run"
                and "snapshot_json" in instance_config
            ):
                raise ValueError(
                    "whole_run snapshot restore is not supported; do not pass snapshot_json"
                )
            request = self._build_start_instance(self._next_request_seq, instance_config)
            response = await self._execute(request, deadline=deadline)
            try:
                result = self._accept_start_instance_for_request(request, response)
            except ApiProtocolError:
                await self._mark_protocol_uncertain(request)
                raise
            self._consume_sequence(RetryRequest.from_message(request))
            return result

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
            try:
                result = self._accept_get_decision(response, branch_id)
            except ApiProtocolError:
                await self._mark_protocol_uncertain(request)
                raise
            self._consume_sequence(RetryRequest.from_message(request))
            return result

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

    async def emulate_actions(
        self,
        instance_id: str,
        items: Sequence[Mapping[str, object]],
        *,
        timeout_s: float,
        simulation_options: Mapping[str, object] | None = None,
    ) -> JsonObject:
        """DTO v0.7 batch counterpart of ``emulate_action``.

        Every item parent must already exist and be usable when the batch request starts;
        a Branch created by another item in the same batch is not a valid parent. The
        whole batch is sent as one request over the existing single in-flight,
        session-sequenced protocol and exact-replayed as a whole if completion is
        uncertain.
        """
        async with self._operation_deadline(timeout_s) as deadline:
            self._ensure_fresh_request_allowed()
            request = self._build_emulate_actions(
                self._next_request_seq, instance_id, items, simulation_options
            )
            return await self._execute_emulate_actions(request, deadline=deadline)

    async def _execute_emulate_actions(
        self,
        request: JsonObject,
        *,
        deadline: float,
    ) -> JsonObject:
        response: JsonObject | None = None
        token = RetryRequest.from_message(request)
        try:
            response = await self._execute(request, deadline=deadline)
            self._consume_sequence(token)
        except asyncio.CancelledError as exc:
            self._record_emulate_actions_batch(request, response, error=exc)
            raise
        except ApiOperationError as exc:
            self._record_emulate_actions_batch(request, exc.response)
            raise
        except Exception as exc:
            self._record_emulate_actions_batch(request, response, error=exc)
            raise

        self._record_emulate_actions_batch(request, response)
        return response

    def _record_emulate_actions_batch(
        self,
        request: Mapping[str, object],
        response: Mapping[str, object] | None,
        *,
        error: BaseException | None = None,
    ) -> None:
        items = request.get("items")
        if not isinstance(items, list):
            return
        branch_results = response.get("branch_results") if response is not None else None
        for item in items:
            if not isinstance(item, Mapping):
                continue
            branch_id = item.get("branch_id")
            branch_result = (
                branch_results.get(branch_id)
                if isinstance(branch_results, Mapping)
                else response
            )
            item_request = {
                "schema_version": request.get("schema_version"),
                "client_session_id": request.get("client_session_id"),
                "request_seq": request.get("request_seq"),
                "request_id": request.get("request_id"),
                "operation": "emulate_actions",
                "instance_id": request.get("instance_id"),
                "parent_branch_id": item.get("parent_branch_id"),
                "branch_id": branch_id,
                "rng_id": item.get("rng_id"),
                "decision_point_id": item.get("decision_point_id"),
                "action_id": item.get("action_id"),
            }
            self._record_selected_action(
                item_request,
                source_branch_id=item.get("parent_branch_id"),
                result=branch_result if isinstance(branch_result, Mapping) else None,
                error=error,
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
            try:
                result = self._accept_close_instance(response)
            except ApiProtocolError:
                await self._mark_protocol_uncertain(request)
                raise
            self._consume_sequence(RetryRequest.from_message(request))
            self._clear_instance_extensions()
            return result

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
            response = await self._execute(request, deadline=deadline)
            self._consume_sequence(RetryRequest.from_message(request))
            return response

    async def _complete_deferred_branch_cleanup(
        self,
        source_retry: RetryRequest,
        *,
        deadline: float,
        include_retry_created: bool = True,
    ) -> None:
        record = self._pending_retry_cleanup
        if record is None or record[0] != source_retry:
            return

        _, instance_id, recorded_branch_ids = record
        if include_retry_created:
            branch_ids = recorded_branch_ids
        else:
            retry_created = self._branch_ids_created_by_retry(source_retry)
            branch_ids = tuple(
                branch_id
                for branch_id in recorded_branch_ids
                if branch_id not in retry_created
            )

        if not branch_ids:
            self._pending_retry_cleanup = None
            return

        cleanup_request = self._build_branch_batch_operation(
            self._next_request_seq,
            "release_branches",
            instance_id,
            branch_ids,
        )
        cleanup_token = RetryRequest.from_message(cleanup_request)
        self._pending_retry_cleanup = (cleanup_token, instance_id, branch_ids)

        # Claim the cleanup sequence before the first cancellation point. This is safe
        # even if the frame has not yet left the process: exact replaying an idempotent
        # release with the same next sequence is valid in either case, while forgetting
        # the cleanup token could leak Branch ownership after a cancellation.
        self._remember_pending(cleanup_token)
        try:
            await self._execute(cleanup_request, deadline=deadline)
            self._consume_sequence(cleanup_token)
        except BaseException:
            if self._pending_retry != cleanup_token:
                self._pending_retry_cleanup = None
            raise
        self._pending_retry_cleanup = None

    @staticmethod
    def _branch_ids_created_by_retry(retry_request: RetryRequest) -> frozenset[str]:
        request = retry_request.to_message()
        if retry_request.operation == "emulate_action":
            branch_id = request.get("branch_id")
            return (
                frozenset({branch_id})
                if isinstance(branch_id, str) and branch_id
                else frozenset()
            )
        if retry_request.operation == "emulate_actions":
            items = request.get("items")
            if not isinstance(items, list):
                return frozenset()
            return frozenset(
                branch_id
                for item in items
                if isinstance(item, Mapping)
                for branch_id in [item.get("branch_id")]
                if isinstance(branch_id, str) and branch_id
            )
        return frozenset()

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
            self._pending_retry = None
            self._pending_retry_cleanup = None
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
            return self._validate_api_response(request, response)
        except ApiOperationError as exc:
            fault_kind = exc.response.get("fault_kind")
            if (
                exc.response.get("status") == "rejected"
                and fault_kind in _NON_CONSUMING_SESSION_REJECTIONS
            ):
                # RL rejected this before admitting it as the session's next executable
                # sequence. Continuing by guessing another seq would desynchronize the
                # state machine, so this logical client is permanently failed closed.
                self._session_invalid = True
                self._pending_retry = None
                self._pending_retry_cleanup = None
                raise

            self._consume_sequence(token)
            if token.operation == "close_instance" and exc.response.get("status") == "faulted":
                self._instance_id = None
                self._instance_type = None
                self._pending_start_instance_type = None
                self._max_emulate_actions_items = None
                self._clear_instance_extensions()
                self._audit.clear()
            if fault_kind in _FATAL_CONSUMING_SESSION_REJECTIONS:
                self._session_invalid = True
                self._pending_retry_cleanup = None
            raise
        except ApiProtocolError:
            await self._mark_protocol_uncertain(request)
            raise

    async def _execute_selected_action(
        self,
        request: JsonObject,
        *,
        source_branch_id: str,
        deadline: float,
    ) -> JsonObject:
        response: JsonObject | None = None
        token = RetryRequest.from_message(request)
        try:
            response = await self._execute(request, deadline=deadline)
            try:
                self._validate_selected_action_response(request, response)
            except ApiProtocolError:
                await self._mark_protocol_uncertain(request)
                raise
            self._consume_sequence(token)
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

    async def _mark_protocol_uncertain(self, request: Mapping[str, object]) -> None:
        # Preserve the exact recovery token before the first cancellation point. RL may
        # already have consumed this request, so connection teardown must not be able to
        # erase the only exact-replay path.
        self._remember_pending(RetryRequest.from_message(request))
        await self._connection.invalidate()

    def _ensure_fresh_request_allowed(self) -> None:
        if self._session_invalid:
            raise RuntimeError("RL session is invalid; create a new client session")
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

    def _clear_instance_extensions(self) -> None:
        self._emulate_actions_boundaries = frozenset()
        self._pending_retry_cleanup = None

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
