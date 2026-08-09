from __future__ import annotations

import unittest

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.api.transport import RetryRequest


class _FakeConnection:
    client_session_id = "session-a"

    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def exchange(self, request: dict, *, deadline: float) -> dict:
        self.requests.append(dict(request))
        return {
            "schema_version": "0.7",
            "server_epoch": "epoch-1",
            "client_session_id": request["client_session_id"],
            "request_seq": request["request_seq"],
            "request_id": request["request_id"],
            "operation": request["operation"],
            "status": "completed",
            "instance_id": "inst-001",
        }

    async def invalidate(self) -> None:
        pass

    async def close(self) -> None:
        pass


class StartInstanceSnapshotGuardTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _snapshot_config(snapshot_json: object = "{...}") -> dict[str, object]:
        return {
            "instance_type": "whole_run",
            "character_id": "IRONCLAD",
            "ascension": 0,
            "seed": 1,
            "snapshot_json": snapshot_json,
        }

    async def test_whole_run_snapshot_is_rejected_before_io_or_sequence_consumption(self) -> None:
        connection = _FakeConnection()
        client = AsyncTrainingApiClient(connection)  # type: ignore[arg-type]

        for snapshot_json in ("{...}", "", None):
            with self.subTest(snapshot_json=snapshot_json):
                with self.assertRaisesRegex(ValueError, "snapshot restore is not supported"):
                    await client.start_instance(
                        self._snapshot_config(snapshot_json),
                        timeout_s=1.0,
                    )

                self.assertEqual(connection.requests, [])
                self.assertEqual(client.next_request_seq, 1)
                self.assertIsNone(client.pending_retry)
                self.assertIsNone(client.instance_id)

    async def test_invalid_timeout_precedes_snapshot_guard(self) -> None:
        connection = _FakeConnection()
        client = AsyncTrainingApiClient(connection)  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "timeout_s must be positive"):
            await client.start_instance(self._snapshot_config(), timeout_s=0)

        self.assertEqual(connection.requests, [])
        self.assertEqual(client.next_request_seq, 1)

    async def test_pending_retry_precedes_snapshot_guard(self) -> None:
        connection = _FakeConnection()
        client = AsyncTrainingApiClient(connection)  # type: ignore[arg-type]
        client._pending_retry = RetryRequest.from_message(  # type: ignore[attr-defined]
            {
                "client_session_id": client.client_session_id,
                "request_seq": 1,
                "request_id": f"{client.client_session_id}:1",
                "operation": "get_decision",
            }
        )

        with self.assertRaisesRegex(RuntimeError, "unresolved request is pending"):
            await client.start_instance(self._snapshot_config(), timeout_s=1.0)

        self.assertEqual(connection.requests, [])
        self.assertEqual(client.next_request_seq, 1)
        self.assertIsNotNone(client.pending_retry)

    async def test_active_instance_precedes_snapshot_guard(self) -> None:
        connection = _FakeConnection()
        client = AsyncTrainingApiClient(connection)  # type: ignore[arg-type]
        await client.start_instance(
            {
                "instance_type": "whole_run",
                "character_id": "IRONCLAD",
                "ascension": 0,
                "seed": 1,
            },
            timeout_s=1.0,
        )

        with self.assertRaisesRegex(RuntimeError, "already has an active instance"):
            await client.start_instance(self._snapshot_config(), timeout_s=1.0)

        self.assertEqual(len(connection.requests), 1)
        self.assertEqual(client.next_request_seq, 2)
        self.assertEqual(client.instance_id, "inst-001")

    async def test_fresh_whole_run_without_snapshot_is_still_supported(self) -> None:
        connection = _FakeConnection()
        client = AsyncTrainingApiClient(connection)  # type: ignore[arg-type]

        instance_id = await client.start_instance(
            {
                "instance_type": "whole_run",
                "character_id": "IRONCLAD",
                "ascension": 0,
                "seed": 1,
            },
            timeout_s=1.0,
        )

        self.assertEqual(instance_id, "inst-001")
        self.assertEqual(len(connection.requests), 1)
        self.assertNotIn("snapshot_json", connection.requests[0]["instance_config"])
        self.assertEqual(client.next_request_seq, 2)


if __name__ == "__main__":
    unittest.main()
