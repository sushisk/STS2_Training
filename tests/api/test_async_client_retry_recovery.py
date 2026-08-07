from __future__ import annotations

import asyncio
import unittest

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.api.transport import RetryRequest, TransportError


class _RetryingConnection:
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.fail_next_commit = True
        self.fail_next_start = False

    async def exchange(self, message, *, deadline: float):
        request = dict(message)
        self.messages.append(request)
        operation = request["operation"]

        if operation == "start_instance":
            if self.fail_next_start:
                self.fail_next_start = False
                raise TransportError(
                    "lost start response",
                    completion_uncertain=True,
                    retry_request=RetryRequest.from_message(request),
                )
            return {
                "schema_version": request["schema_version"],
                "request_id": request["request_id"],
                "operation": operation,
                "status": "completed",
                "instance_id": "inst-001",
                "decision_point_id": "decision-1",
                "masked_emulator_dto": {"state": "initial"},
            }

        if operation == "commit_action":
            if self.fail_next_commit:
                self.fail_next_commit = False
                raise TransportError(
                    "lost commit response",
                    completion_uncertain=True,
                    retry_request=RetryRequest.from_message(request),
                )
            return {
                "schema_version": request["schema_version"],
                "request_id": request["request_id"],
                "operation": operation,
                "status": "completed",
                "instance_id": request["instance_id"],
                "branch_id": "root",
                "decision_point_id": "decision-2",
                "masked_emulator_dto": {"state": "after"},
            }

        if operation == "get_decision":
            return {
                "schema_version": request["schema_version"],
                "request_id": request["request_id"],
                "operation": operation,
                "status": "completed",
                "instance_id": request["instance_id"],
                "branch_id": request["branch_id"],
                "decision_point_id": "decision-1",
                "masked_emulator_dto": {"state": "decision"},
            }

        raise AssertionError(f"unexpected operation: {operation}")

    async def invalidate(self) -> None:
        pass

    async def close(self) -> None:
        pass


class _CancelledCommitConnection:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def exchange(self, message, *, deadline: float):
        request = dict(message)
        self.messages.append(request)
        if request["operation"] == "start_instance":
            return {
                "schema_version": request["schema_version"],
                "request_id": request["request_id"],
                "operation": request["operation"],
                "status": "completed",
                "instance_id": "inst-001",
                "decision_point_id": "decision-1",
                "masked_emulator_dto": {"state": "initial"},
            }

        exc = asyncio.CancelledError()
        setattr(exc, "completion_uncertain", True)
        setattr(exc, "retry_request", RetryRequest.from_message(request))
        raise exc

    async def invalidate(self) -> None:
        pass

    async def close(self) -> None:
        pass


class AsyncClientRetryRecoveryTest(unittest.IsolatedAsyncioTestCase):
    async def test_uncertain_commit_exposes_same_id_retry_and_blocks_fresh_calls(self) -> None:
        connection = _RetryingConnection()
        client = AsyncTrainingApiClient(connection)
        instance_id = await client.start_instance(
            {"instance_type": "combat"},
            timeout_s=1.0,
        )

        with self.assertRaisesRegex(TransportError, "lost commit response"):
            await client.commit_action(
                instance_id,
                "decision-1",
                "action-1",
                timeout_s=1.0,
            )

        retry = client.pending_retry
        self.assertIsNotNone(retry)
        assert retry is not None
        original_commit = connection.messages[-1]
        self.assertEqual(retry.request_id, original_commit["request_id"])

        with self.assertRaisesRegex(RuntimeError, "completion-uncertain"):
            await client.get_decision(instance_id, timeout_s=1.0)
        self.assertEqual(len(connection.messages), 2)

        response = await client.retry_request(retry, timeout_s=1.0)
        self.assertIsInstance(response, dict)
        self.assertEqual(response["status"], "completed")
        self.assertIsNone(client.pending_retry)
        replayed_commit = connection.messages[-1]
        self.assertEqual(replayed_commit, original_commit)

    async def test_uncertain_start_can_be_recovered_with_retry_token(self) -> None:
        connection = _RetryingConnection()
        connection.fail_next_start = True
        client = AsyncTrainingApiClient(connection)

        with self.assertRaisesRegex(TransportError, "lost start response"):
            await client.start_instance(
                {"instance_type": "combat"},
                timeout_s=1.0,
            )

        retry = client.pending_retry
        self.assertIsNotNone(retry)
        self.assertTrue(client.start_uncertain)
        assert retry is not None

        instance_id = await client.retry_request(retry, timeout_s=1.0)
        self.assertEqual(instance_id, "inst-001")
        self.assertFalse(client.start_uncertain)
        self.assertIsNone(client.pending_retry)
        self.assertEqual(connection.messages[0], connection.messages[1])

    async def test_cancelled_selection_is_recorded_before_reraise(self) -> None:
        events: list[dict] = []
        connection = _CancelledCommitConnection()
        client = AsyncTrainingApiClient(connection, selection_logger=events.append)
        instance_id = await client.start_instance(
            {"instance_type": "combat"},
            timeout_s=1.0,
        )

        with self.assertRaises(asyncio.CancelledError):
            await client.commit_action(
                instance_id,
                "decision-1",
                "action-1",
                timeout_s=1.0,
            )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["event"], "selection")
        self.assertEqual(event["request"]["operation"], "commit_action")
        self.assertEqual(event["client_error"]["type"], "CancelledError")
        self.assertIsNotNone(client.pending_retry)


if __name__ == "__main__":
    unittest.main()
