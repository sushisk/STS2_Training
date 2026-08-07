from __future__ import annotations

import asyncio
import json
import unittest

from sts2_training.api.asyncio_tcp_transport import AsyncioTcpTransport
from sts2_training.api.transport import TransportError


class AsyncioTcpTransportTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.requests: list[dict] = []
        self.connection_count = 0
        self.server = await asyncio.start_server(
            self._handle_client,
            "127.0.0.1",
            0,
        )
        self.port = int(self.server.sockets[0].getsockname()[1])
        self.transport = AsyncioTcpTransport(port=self.port)

    async def asyncTearDown(self) -> None:
        await self.transport.close()
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
                response = (
                    {"transport_operation": "pong"}
                    if request.get("transport_operation") == "ping"
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
            await self.transport.ping(timeout_s=1.0),
            {"transport_operation": "pong"},
        )

    async def test_concurrent_calls_share_connection(self) -> None:
        first, second = await asyncio.gather(
            self.transport.call({"request_id": "1"}, timeout_s=1.0),
            self.transport.call({"request_id": "2"}, timeout_s=1.0),
        )
        self.assertEqual(first, {"echo": {"request_id": "1"}})
        self.assertEqual(second, {"echo": {"request_id": "2"}})
        self.assertEqual(self.connection_count, 1)

    async def test_timeout_discards_connection(self) -> None:
        with self.assertRaisesRegex(TransportError, "timed out"):
            await self.transport.call({"request_id": "slow"}, timeout_s=0.01)
        self.assertFalse(self.transport.is_alive())

        response = await self.transport.call(
            {"request_id": "next"}, timeout_s=1.0
        )
        self.assertEqual(response, {"echo": {"request_id": "next"}})
        self.assertEqual(self.connection_count, 2)


if __name__ == "__main__":
    unittest.main()
