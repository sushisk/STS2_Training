from __future__ import annotations

import asyncio
import unittest

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.api.transport import (
    RetryRequest,
    ServerEpochChangedError,
    TransportError,
)


class _Connection:
    client_session_id = "session-a"

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.fail_operation_once: str | None = None
        self.cancel_operation_once: str | None = None
        self.change_epoch_on_next = False

    @staticmethod
    def _response(request: dict, **fields) -> dict:
        return {
            "schema_version": "0.6",
            "server_epoch": "epoch-1",
            "client_session_id": request["client_session_id"],
            "request_seq": request["request_seq"],
            "request_id": request["request_id"],
            "operation": request["operation"],
            **fields,
        }

    async def exchange(self, message, *, deadline: float):
        request = dict(message)
        self.messages.append(request)
        operation = request["operation"]

        if self.change_epoch_on_next:
            self.change_epoch_on_next = False
            raise ServerEpochChangedError(
                expected_epoch="epoch-1",
                actual_epoch="epoch-2",
            )

        if self.fail_operation_once == operation:
            self.fail_operation_once = None
            raise TransportError(
                f"lost {operation} response",
                completion_uncertain=True,
                retry_request=RetryRequest.from_message(request),
            )

        if self.cancel_operation_once == operation:
            self.cancel_operation_once = None
            exc = asyncio.CancelledError()
            setattr(exc, "completion_uncertain", True)
            setattr(exc, "retry_request", RetryRequest.from_message(request))
            raise exc

        if operation == "start_instance":
            return self._response(
                request,
                status="completed",
                instance_id="inst-001",
                decision_point_id="decision-1",
                masked_emulator_dto={"state": "initial"},
            )
        if operation == "get_decision":
            return self._response(
                request,
                status="completed",
                instance_id=request["instance_id"],
                branch_id=request["branch_id"],
                decision_point_id="decision-1",
                masked_emulator_dto={"state": "decision"},
            )
        if operation == "commit_action":
            return self._response(
                request,
                status="completed",
                instance_id=request["instance_id"],
                branch_id="root",
                decision_point_id="decision-2",
                masked_emulator_dto={"state": "after"},
            )
        if operation == "close_instance":
            return self._response(
                request,
                status="completed",
                instance_id=request["instance_id"],
            )
        raise AssertionError(f"unexpected operation: {operation}")

    async def invalidate(self) -> None:
        pass

    async def close(self) -> None:
        pass


class AsyncClientRetryRecoveryTest(unittest.IsolatedAsyncioTestCase):
    async def test_uncertain_commit_keeps_same_sequence_and_blocks_fresh_calls(self) -> None:
        connection = _Connection()
        client = AsyncTrainingApiClient(connection)
        instance_id = await client.start_instance(
            {"instance_type": "combat"}, timeout_s=1.0
        )
        self.assertEqual(client.next_request_seq, 2)

        connection.fail_operation_once = "commit_action"
        with self.assertRaisesRegex(TransportError, "lost commit_action response"):
            await client.commit_action(
                instance_id,
                "decision-1",
                "action-1",
                timeout_s=1.0,
            )

        retry = client.pending_retry
        self.assertIsNotNone(retry)
        assert retry is not None
        self.assertEqual(retry.request_seq, 2)
        original = connection.messages[-1]
        self.assertEqual(retry.to_message(), original)
        self.assertEqual(client.next_request_seq, 2)

        with self.assertRaisesRegex(RuntimeError, "unresolved request"):
            await client.get_decision(instance_id, timeout_s=1.0)

        response = await client.retry_request(retry, timeout_s=1.0)
        self.assertEqual(response["status"], "completed")
        self.assertEqual(connection.messages[-1], original)
        self.assertIsNone(client.pending_retry)
        self.assertEqual(client.next_request_seq, 3)

    async def test_read_only_request_uses_same_uncertainty_rule(self) -> None:
        connection = _Connection()
        client = AsyncTrainingApiClient(connection)
        instance_id = await client.start_instance(
            {"instance_type": "combat"}, timeout_s=1.0
        )

        connection.fail_operation_once = "get_decision"
        with self.assertRaises(TransportError):
            await client.get_decision(instance_id, timeout_s=1.0)

        retry = client.pending_retry
        self.assertIsNotNone(retry)
        assert retry is not None
        self.assertEqual(retry.request_seq, 2)
        decision = await client.retry_request(retry, timeout_s=1.0)
        self.assertEqual(decision["decision_point_id"], "decision-1")
        self.assertEqual(client.next_request_seq, 3)

    async def test_uncertain_start_is_recovered_by_exact_sequence_replay(self) -> None:
        connection = _Connection()
        connection.fail_operation_once = "start_instance"
        client = AsyncTrainingApiClient(connection)

        with self.assertRaises(TransportError):
            await client.start_instance(
                {"instance_type": "combat"}, timeout_s=1.0
            )

        retry = client.pending_retry
        self.assertTrue(client.start_uncertain)
        assert retry is not None
        instance_id = await client.retry_request(retry, timeout_s=1.0)
        self.assertEqual(instance_id, "inst-001")
        self.assertFalse(client.start_uncertain)
        self.assertEqual(connection.messages[0], connection.messages[1])
        self.assertEqual(client.next_request_seq, 2)

    async def test_epoch_change_permanently_invalidates_client_session(self) -> None:
        connection = _Connection()
        client = AsyncTrainingApiClient(connection)
        instance_id = await client.start_instance(
            {"instance_type": "combat"}, timeout_s=1.0
        )
        connection.change_epoch_on_next = True

        with self.assertRaises(ServerEpochChangedError):
            await client.get_decision(instance_id, timeout_s=1.0)

        self.assertTrue(client.session_invalid)
        with self.assertRaisesRegex(RuntimeError, "epoch changed"):
            await client.get_decision(instance_id, timeout_s=1.0)

    async def test_cancelled_selection_is_a_pending_retry_and_is_audited(self) -> None:
        events: list[dict] = []
        connection = _Connection()
        client = AsyncTrainingApiClient(connection, selection_logger=events.append)
        instance_id = await client.start_instance(
            {"instance_type": "combat"}, timeout_s=1.0
        )
        connection.cancel_operation_once = "commit_action"

        with self.assertRaises(asyncio.CancelledError):
            await client.commit_action(
                instance_id,
                "decision-1",
                "action-1",
                timeout_s=1.0,
            )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "selection")
        self.assertEqual(events[0]["client_error"]["type"], "CancelledError")
        self.assertEqual(client.pending_retry.request_seq, 2)


if __name__ == "__main__":
    unittest.main()
