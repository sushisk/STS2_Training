from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

from sts2_training.api.tcp_connection import TcpConnection
from sts2_training.api.transport import ServerEpochChangedError, TransportError


class TcpConnectionTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.requests: list[dict] = []
        self.connection_count = 0
        self.cancel_received = asyncio.Event()
        self.release_cancel = asyncio.Event()
        self.delay_ping = False
        self.epoch = "epoch-1"
        self.server = await asyncio.start_server(self._handle_client, "127.0.0.1", 0)
        self.port = int(self.server.sockets[0].getsockname()[1])
        self.connection = TcpConnection(
            port=self.port,
            client_session_id="session-a",
        )

    async def asyncTearDown(self) -> None:
        self.release_cancel.set()
        await self.connection.close()
        self.server.close()
        await self.server.wait_closed()

    def _message(self, request_id: str, **fields) -> dict:
        return {
            "client_session_id": "session-a",
            "request_id": request_id,
            **fields,
        }

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.connection_count += 1
        try:
            while line := await reader.readline():
                request = json.loads(line)
                if request.get("transport_operation") == "hello":
                    response = {
                        "transport_operation": "hello",
                        "schema_version": "0.6",
                        "client_session_id": request["client_session_id"],
                        "server_epoch": self.epoch,
                    }
                elif request == {"transport_operation": "ping"}:
                    if self.delay_ping:
                        await asyncio.sleep(0.05)
                    response = {
                        "transport_operation": "pong",
                        "server_epoch": self.epoch,
                    }
                else:
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
                    elif request.get("request_id") == "large-response":
                        response = {
                            "server_epoch": self.epoch,
                            "payload": "x" * 4096,
                        }
                    else:
                        response = {
                            "server_epoch": self.epoch,
                            "echo": request,
                        }
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

    async def test_connect_performs_hello_and_ping_checks_epoch(self) -> None:
        response = await self.connection.ping(timeout_s=1.0)
        self.assertEqual(
            response,
            {"transport_operation": "pong", "server_epoch": "epoch-1"},
        )
        self.assertEqual(self.connection.server_epoch, "epoch-1")
        self.assertEqual(self.requests, [])

    async def test_ping_timeout_discards_stream_before_next_exchange(self) -> None:
        self.delay_ping = True
        with self.assertRaisesRegex(TransportError, "ping timed out"):
            await self.connection.ping(timeout_s=0.01)
        self.assertFalse(self.connection.is_alive())

        self.delay_ping = False
        response = await self.connection.exchange(
            self._message("after-ping-timeout"), timeout_s=1.0
        )
        self.assertEqual(response["echo"], self._message("after-ping-timeout"))
        self.assertEqual(self.connection_count, 2)

    async def test_concurrent_exchanges_are_serialized_on_one_connection(self) -> None:
        first, second = await asyncio.gather(
            self.connection.exchange(self._message("1"), timeout_s=1.0),
            self.connection.exchange(self._message("2"), timeout_s=1.0),
        )
        self.assertEqual(first["echo"], self._message("1"))
        self.assertEqual(second["echo"], self._message("2"))
        self.assertEqual(self.connection_count, 1)

    async def test_connection_lock_wait_consumes_exchange_timeout(self) -> None:
        await self.connection._lock.acquire()
        try:
            with self.assertRaisesRegex(TransportError, "timed out") as caught:
                await self.connection.exchange(
                    self._message("queued"), timeout_s=0.01
                )
        finally:
            self.connection._lock.release()
        self.assertFalse(caught.exception.completion_uncertain)
        self.assertIsNone(caught.exception.retry_request)
        self.assertEqual(self.requests, [])

    async def test_connect_consumes_exchange_timeout(self) -> None:
        connection = TcpConnection(
            port=self.port,
            client_session_id="session-a",
            connect_timeout_s=5.0,
        )

        async def slow_open_connection(*args, **kwargs):
            await asyncio.sleep(1.0)
            raise AssertionError("exchange deadline should expire first")

        try:
            with patch(
                "sts2_training.api.tcp_connection.asyncio.open_connection",
                side_effect=slow_open_connection,
            ):
                with self.assertRaises(TransportError) as caught:
                    await connection.exchange(
                        self._message("connect-slow"), timeout_s=0.01
                    )
            self.assertFalse(caught.exception.completion_uncertain)
        finally:
            await connection.close()

    async def test_timeout_after_send_is_completion_uncertain(self) -> None:
        with self.assertRaisesRegex(TransportError, "timed out") as caught:
            await self.connection.exchange(self._message("slow"), timeout_s=0.01)
        self.assertTrue(caught.exception.completion_uncertain)
        self.assertEqual(
            caught.exception.retry_request.to_message(), self._message("slow")
        )
        self.assertFalse(self.connection.is_alive())

        response = await self.connection.exchange(
            self._message("next"), timeout_s=1.0
        )
        self.assertEqual(response["echo"], self._message("next"))
        self.assertEqual(self.connection_count, 2)

    async def test_external_cancellation_after_send_marks_completion_uncertain(self) -> None:
        task = asyncio.create_task(
            self.connection.exchange(self._message("cancel"), timeout_s=1.0)
        )
        await asyncio.wait_for(self.cancel_received.wait(), timeout=1.0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError) as caught:
            await task
        self.assertTrue(getattr(caught.exception, "completion_uncertain", False))
        retry_request = getattr(caught.exception, "retry_request", None)
        self.assertEqual(retry_request.to_message(), self._message("cancel"))
        self.assertFalse(self.connection.is_alive())

    async def test_request_too_large_is_known_pre_send_failure(self) -> None:
        connection = TcpConnection(
            port=self.port,
            client_session_id="session-a",
            max_message_bytes=64,
        )
        try:
            with self.assertRaisesRegex(TransportError, "request exceeds") as caught:
                await connection.exchange(
                    self._message("large", payload="x" * 128),
                    timeout_s=1.0,
                )
            self.assertFalse(caught.exception.completion_uncertain)
            self.assertEqual(self.requests, [])
        finally:
            await connection.close()

    async def test_transport_error_response_is_definitive_failure(self) -> None:
        with self.assertRaisesRegex(TransportError, "message_too_large") as caught:
            await self.connection.exchange(
                self._message("transport-error"), timeout_s=1.0
            )
        self.assertFalse(caught.exception.completion_uncertain)
        self.assertIsNone(caught.exception.retry_request)
        self.assertFalse(self.connection.is_alive())

    async def test_response_limit_is_completion_uncertain_and_can_be_raised(self) -> None:
        connection = TcpConnection(
            port=self.port,
            client_session_id="session-a",
            max_response_bytes=1024,
        )
        try:
            with self.assertRaisesRegex(
                TransportError, "response exceeds max_response_bytes=1024"
            ) as caught:
                await connection.exchange(
                    self._message("large-response"), timeout_s=1.0
                )
            self.assertTrue(caught.exception.completion_uncertain)
            token = caught.exception.retry_request
            self.assertIsNotNone(token)

            await connection.set_max_response_bytes(8192)
            response = await connection.exchange(token.to_message(), timeout_s=1.0)
            self.assertEqual(response["payload"], "x" * 4096)
        finally:
            await connection.close()

    async def test_epoch_change_is_detected_before_reconnect_retry_send(self) -> None:
        await self.connection.ping(timeout_s=1.0)
        await self.connection.invalidate()
        self.epoch = "epoch-2"

        with self.assertRaises(ServerEpochChangedError) as caught:
            await self.connection.exchange(self._message("after-restart"), timeout_s=1.0)

        self.assertEqual(caught.exception.expected_epoch, "epoch-1")
        self.assertEqual(caught.exception.actual_epoch, "epoch-2")
        self.assertFalse(caught.exception.completion_uncertain)
        self.assertNotIn(self._message("after-restart"), self.requests)


if __name__ == "__main__":
    unittest.main()
