from __future__ import annotations

import asyncio
import json
import unittest

from sts2_training.api.asyncio_tcp_transport import AsyncioTcpTransport


class AsyncioTcpTransportTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.requests: list[dict] = []
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
        try:
            while line := await reader.readline():
                request = json.loads(line)
                self.requests.append(request)
                if request.get("transport_operation") == "ping":
                    response = {"transport_operation": "pong"}
                else:
                    response = {"echo": request}
                writer.write(json.dumps(response).encode("utf-8") + b"\n")
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def test_ping(self) -> None:
        response = await self.transport.ping(timeout_s=1.0)
        self.assertEqual(response, {"transport_operation": "pong"})

    async def test_call_reuses_connection(self) -> None:
        first = await self.transport.call({"request_id": "1"}, timeout_s=1.0)
        second = await self.transport.call({"request_id": "2"}, timeout_s=1.0)
        self.assertEqual(first, {"echo": {"request_id": "1"}})
        self.assertEqual(second, {"echo": {"request_id": "2"}})
        self.assertTrue(self.transport.is_alive())


if __name__ == "__main__":
    unittest.main()
