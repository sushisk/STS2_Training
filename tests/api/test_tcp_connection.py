from __future__ import annotations

import asyncio
import json
import unittest

from sts2_training.api.tcp_connection import TcpConnection
from sts2_training.api.transport import TransportError


class TcpConnectionTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.requests: list[dict] = []
        self.connection_count = 0
        self.cancel_received = asyncio.Event()
        self.release_cancel = asyncio.Event()
        self.server = await asyncio.start_server(
            self._handle_client,
            "127.0.0.1",
            0,
        )
        self.port = int(self.server.sockets[0].getsockname()[1])
        self.connection = TcpConnection(port=self.port)

    async def asyncTearDown(self) -> None:
        self.release_cancel.set()
        await self.connection.close()
        self.server.close()
        await self.server.wait_closed()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.connection_count += 1
        try:
            while line := await reader.readline():
                request = json.loads(line)
                self.requests.append(request)
                if request.get("request_id") == "slow":
                    await asyncio.sleep(0.05)
                if request.get("request_id") == "cancel":
                    self.cancel_received.set()
                    await self.release_cancel.wait()
                if request.get("request_id") == "transport-error":
                    response = {
                        "transport_error": "message_too_large",
                        "max_message_bytes": 128,
                    }
                else:
                    response = (
                        {"transport_operation": "pong"}
                        if request == {"transport_operation": "ping"}
                        else {"echo": request}
                    )
                writer.write(json.dumps(response).encode("utf-8") + b"\n")
                await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def test_ping(self) -> None:
        self.assertEqual(
            await self.connection.ping(timeout_s=1.0),
            {"transport_operation": "pong"},
        )

    async def test_concurrent_exchanges_share_connection(self) -> None:
        first, second = await asyncio.gather(
            self.connection.exchange({"request_id": "1"}, timeout_s=1.0),
            self.connection.exchange({"request_id": "2"}, timeout_s=1.0),
        )
        self.assertEqual(first, {"echo": {"request_id": "1"}})
        self.assertEqual(second, {"echo": {"request_id": "2"}})
        self.assertEqual(self.connection_count, 1)

    async def test_timeout_discards_connection(self) -> None:
        with self.assertRaisesRegex(TransportError, "timed out"):
            await self.connection.exchange(
                {"request_id": "slow"},
                timeout_s=0.01,
            )
        self.assertFalse(self.connection.is_alive())

        response = await self.connection.exchange(
            {"request_id": "next"},
            timeout_s=1.0,
        )
        self.assertEqual(response, {"echo": {"request_id": "next"}})
        self.assertEqual(self.connection_count, 2)

    async def test_external_cancellation_discards_connection_before_next_request(self) -> None:
        task = asyncio.create_task(
            self.connection.exchange(
                {"request_id": "cancel"},
                timeout_s=1.0,
            )
        )
        await asyncio.wait_for(self.cancel_received.wait(), timeout=1.0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.connection.is_alive())

        # Let the old server-side handler produce its late response after the client has
        # already discarded that stream. The next call must use a fresh connection.
        self.release_cancel.set()
        await asyncio.sleep(0)
        response = await self.connection.exchange(
            {"request_id": "after-cancel"},
            timeout_s=1.0,
        )
        self.assertEqual(response, {"echo": {"request_id": "after-cancel"}})
        self.assertEqual(self.connection_count, 2)

    async def test_transport_error_response_is_transport_error_and_disconnects(self) -> None:
        with self.assertRaisesRegex(TransportError, "message_too_large"):
            await self.connection.exchange(
                {"request_id": "transport-error"},
                timeout_s=1.0,
            )
        self.assertFalse(self.connection.is_alive())


if __name__ == "__main__":
    unittest.main()
