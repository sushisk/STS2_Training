from __future__ import annotations

import unittest

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.api.contract import ApiProtocolError


class _BranchBatchConnection:
    def __init__(self, operation: str, malformed_branch_statuses: object) -> None:
        self.client_session_id = "session-a"
        self.operation = operation
        self.malformed_branch_statuses = malformed_branch_statuses
        self.branch_attempts = 0
        self.invalidations = 0

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def invalidate(self) -> None:
        self.invalidations += 1

    async def exchange(self, message: dict, *, deadline: float) -> dict:
        common = {
            "schema_version": "0.7",
            "server_epoch": "epoch-1",
            "client_session_id": message["client_session_id"],
            "request_seq": message["request_seq"],
            "request_id": message["request_id"],
            "operation": message["operation"],
        }
        if message["operation"] == "start_instance":
            return {
                **common,
                "status": "completed",
                "instance_id": "inst-001",
            }

        self.branch_attempts += 1
        if self.branch_attempts == 1:
            branch_statuses = self.malformed_branch_statuses
        elif self.operation == "cancel_branches":
            branch_statuses = {branch_id: "cancelled" for branch_id in message["branch_ids"]}
        elif self.operation == "release_branches":
            branch_statuses = {branch_id: "released" for branch_id in message["branch_ids"]}
        else:
            branch_statuses = {branch_id: "completed" for branch_id in message["branch_ids"]}

        return {
            **common,
            "status": "completed",
            "instance_id": message["instance_id"],
            "branch_statuses": branch_statuses,
        }


class BranchBatchResponseValidationTest(unittest.IsolatedAsyncioTestCase):
    async def _started_client(
        self,
        operation: str,
        malformed_branch_statuses: object,
    ) -> tuple[AsyncTrainingApiClient, _BranchBatchConnection, str]:
        connection = _BranchBatchConnection(operation, malformed_branch_statuses)
        client = AsyncTrainingApiClient(connection)  # type: ignore[arg-type]
        instance_id = await client.start_instance(
            {"instance_type": "combat"}, timeout_s=1.0
        )
        return client, connection, instance_id

    async def test_cancel_malformed_status_does_not_consume_sequence_and_retry_recovers(self) -> None:
        client, connection, instance_id = await self._started_client(
            "cancel_branches",
            {"branch-a": "running", "branch-b": "cancelled"},
        )
        try:
            with self.assertRaisesRegex(ApiProtocolError, "invalid branch status"):
                await client.cancel_branches(
                    instance_id, ["branch-a", "branch-b"], timeout_s=1.0
                )

            self.assertEqual(client.next_request_seq, 2)
            self.assertIsNotNone(client.pending_retry)
            self.assertEqual(client.pending_retry.request_seq, 2)
            self.assertEqual(connection.invalidations, 1)

            recovered = await client.retry_request(client.pending_retry, timeout_s=1.0)
            self.assertEqual(
                recovered["branch_statuses"],
                {"branch-a": "cancelled", "branch-b": "cancelled"},
            )
            self.assertEqual(client.next_request_seq, 3)
            self.assertIsNone(client.pending_retry)
        finally:
            await client.close()

    async def test_release_missing_branch_status_keeps_sequence_unresolved(self) -> None:
        client, connection, instance_id = await self._started_client(
            "release_branches",
            {"branch-a": "released"},
        )
        try:
            with self.assertRaisesRegex(ApiProtocolError, "keys do not match"):
                await client.release_branches(
                    instance_id, ["branch-a", "branch-b"], timeout_s=1.0
                )

            self.assertEqual(client.next_request_seq, 2)
            self.assertIsNotNone(client.pending_retry)
            self.assertEqual(connection.invalidations, 1)
        finally:
            await client.close()

    async def test_get_branch_status_rejects_request_only_status_value(self) -> None:
        client, connection, instance_id = await self._started_client(
            "get_branch_status",
            {"branch-a": "rejected"},
        )
        try:
            with self.assertRaisesRegex(ApiProtocolError, "invalid branch status"):
                await client.get_branch_status(
                    instance_id, ["branch-a"], timeout_s=1.0
                )

            self.assertEqual(client.next_request_seq, 2)
            self.assertIsNotNone(client.pending_retry)
            self.assertEqual(connection.invalidations, 1)
        finally:
            await client.close()


if __name__ == "__main__":
    unittest.main()
