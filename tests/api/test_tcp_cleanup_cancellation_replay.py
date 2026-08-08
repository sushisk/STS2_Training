from __future__ import annotations

import asyncio
import unittest

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.api.contract import SCHEMA_VERSION
from sts2_training.api.tcp_connection import TcpConnection
from sts2_training.api.transport import RetryRequest


class _Writer:
    def write(self, data: bytes) -> None:
        pass

    async def drain(self) -> None:
        pass


class _CleanupCancelledConnection(TcpConnection):
    def __init__(self) -> None:
        super().__init__(client_session_id="session")
        self.disconnect_calls = 0

    async def _connect(self, *, deadline: float | None = None) -> None:
        self._reader = object()  # type: ignore[assignment]
        self._writer = _Writer()  # type: ignore[assignment]

    async def _read_response_frame(self, reader) -> bytes:
        return b"{not-json}\n"

    async def _disconnect(self) -> None:
        self.disconnect_calls += 1
        raise asyncio.CancelledError()


class TcpCleanupCancellationReplayTest(unittest.IsolatedAsyncioTestCase):
    async def test_decode_cleanup_cancellation_preserves_exact_retry(self) -> None:
        connection = _CleanupCancelledConnection()
        client = AsyncTrainingApiClient(connection)
        request = {
            "schema_version": SCHEMA_VERSION,
            "client_session_id": connection.client_session_id,
            "request_seq": 1,
            "request_id": "req-cleanup-cancel",
            "operation": "get_decision",
            "instance_id": "instance-1",
            "branch_id": "root",
        }
        deadline = asyncio.get_running_loop().time() + 1.0

        with self.assertRaises(asyncio.CancelledError):
            await client._execute(request, deadline=deadline)  # noqa: SLF001

        self.assertGreaterEqual(connection.disconnect_calls, 2)
        self.assertEqual(client.pending_retry, RetryRequest.from_message(request))
        self.assertEqual(client.next_request_seq, 1)
