"""Async TCP connection for the RL/Training session-sequenced DTO contract v0.6."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Mapping
from typing import Any

from sts2_training.api.transport import (
    JsonObject,
    RetryRequest,
    RuntimeExitedError,
    ServerEpochChangedError,
    TransportClosedError,
    TransportError,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_MAX_MESSAGE_BYTES = 1024 * 1024
DEFAULT_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_RESPONSE_READ_CHUNK_BYTES = 64 * 1024


class TcpConnection:
    """Persistent UTF-8 NDJSON connection with an epoch-checked session handshake."""

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        client_session_id: str | None = None,
        connect_timeout_s: float = 5.0,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        if not host:
            raise ValueError("host must not be empty")
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be positive")
        if max_message_bytes <= 0 or max_response_bytes <= 0:
            raise ValueError("message limits must be positive")
        self._host = host
        self._port = port
        self._client_session_id = client_session_id or str(uuid.uuid4())
        if not self._client_session_id:
            raise ValueError("client_session_id must not be empty")
        self._connect_timeout_s = connect_timeout_s
        self._max_message_bytes = max_message_bytes
        self._max_response_bytes = max_response_bytes
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()
        self._closed = False
        self._server_epoch: str | None = None

    @property
    def client_session_id(self) -> str:
        return self._client_session_id

    @property
    def server_epoch(self) -> str | None:
        return self._server_epoch

    @property
    def max_response_bytes(self) -> int:
        return self._max_response_bytes

    async def set_max_response_bytes(self, max_response_bytes: int) -> None:
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        async with self._lock:
            if self._closed:
                raise TransportClosedError("connection is closed")
            self._max_response_bytes = max_response_bytes

    async def connect(self) -> None:
        async with self._lock:
            await self._connect()

    async def exchange(
        self,
        message: Mapping[str, Any],
        *,
        timeout_s: float | None = None,
        deadline: float | None = None,
    ) -> JsonObject:
        deadline = self._resolve_deadline(timeout_s=timeout_s, deadline=deadline)
        retry_request = RetryRequest.from_message(message)
        if message.get("client_session_id") != self._client_session_id:
            raise ValueError("message client_session_id does not match connection session")
        encoded = retry_request.serialized_payload.encode("utf-8") + b"\n"
        if len(encoded) > self._max_message_bytes:
            raise TransportError(
                f"request exceeds max_message_bytes={self._max_message_bytes}"
            )

        sent = False
        try:
            async with asyncio.timeout_at(deadline):
                async with self._lock:
                    await self._connect(deadline=deadline)
                    assert self._reader is not None
                    assert self._writer is not None
                    sent = True
                    self._writer.write(encoded)
                    await self._writer.drain()
                    line = await self._read_response_frame(self._reader)
        except ServerEpochChangedError:
            raise
        except asyncio.CancelledError as exc:
            if sent:
                await self._disconnect()
            setattr(exc, "completion_uncertain", sent)
            setattr(exc, "retry_request", retry_request if sent else None)
            raise
        except TimeoutError as exc:
            if sent:
                await self._disconnect()
            raise TransportError(
                "RL TCP request timed out",
                completion_uncertain=sent,
                retry_request=retry_request if sent else None,
            ) from exc
        except TransportError as exc:
            if sent and not isinstance(exc, ServerEpochChangedError):
                await self._disconnect()
                exc.completion_uncertain = True
                exc.retry_request = retry_request
            raise
        except (ConnectionError, OSError) as exc:
            if sent:
                await self._disconnect()
            raise RuntimeExitedError(
                "RL TCP connection closed during request",
                completion_uncertain=sent,
                retry_request=retry_request if sent else None,
            ) from exc

        response = await self._decode_response(line, retry_request=retry_request)
        response_epoch = response.get("server_epoch")
        if not isinstance(response_epoch, str) or not response_epoch:
            await self._disconnect()
            raise TransportError(
                "RL API response is missing server_epoch",
                completion_uncertain=True,
                retry_request=retry_request,
            )
        self._assert_epoch(
            response_epoch,
            completion_uncertain=True,
            retry_request=retry_request,
        )
        return response

    async def ping(self, *, timeout_s: float = 5.0) -> JsonObject:
        deadline = self._resolve_deadline(timeout_s=timeout_s, deadline=None)
        async with asyncio.timeout_at(deadline):
            async with self._lock:
                await self._connect(deadline=deadline)
                assert self._reader is not None
                assert self._writer is not None
                self._writer.write(b'{"transport_operation":"ping"}\n')
                await self._writer.drain()
                line = await self._read_response_frame(self._reader)
        response = json.loads(line)
        if not isinstance(response, dict) or response.get("transport_operation") != "pong":
            await self.invalidate()
            raise TransportError("RL TCP server returned an invalid ping response")
        epoch = response.get("server_epoch")
        if not isinstance(epoch, str) or not epoch:
            await self.invalidate()
            raise TransportError("RL ping response is missing server_epoch")
        self._assert_epoch(epoch)
        return response

    def is_alive(self) -> bool:
        return (
            not self._closed
            and self._reader is not None
            and self._writer is not None
            and not self._reader.at_eof()
            and not self._writer.is_closing()
        )

    async def invalidate(self) -> None:
        async with self._lock:
            await self._disconnect()

    async def close(self) -> None:
        async with self._lock:
            if not self._closed:
                self._closed = True
                await self._disconnect()

    async def __aenter__(self) -> "TcpConnection":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    @staticmethod
    def _resolve_deadline(*, timeout_s: float | None, deadline: float | None) -> float:
        if (timeout_s is None) == (deadline is None):
            raise ValueError("provide exactly one of timeout_s or deadline")
        loop = asyncio.get_running_loop()
        if deadline is not None:
            return deadline
        assert timeout_s is not None
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        return loop.time() + timeout_s

    async def _connect(self, *, deadline: float | None = None) -> None:
        if self._closed:
            raise TransportClosedError("connection is closed")
        if self.is_alive():
            return

        loop = asyncio.get_running_loop()
        connect_deadline = loop.time() + self._connect_timeout_s
        if deadline is not None:
            connect_deadline = min(connect_deadline, deadline)
        try:
            async with asyncio.timeout_at(connect_deadline):
                reader, writer = await asyncio.open_connection(
                    self._host,
                    self._port,
                    limit=self._max_message_bytes + 1,
                )
                self._reader, self._writer = reader, writer
                await self._hello(reader, writer)
        except ServerEpochChangedError:
            await self._disconnect()
            raise
        except TimeoutError as exc:
            await self._disconnect()
            raise TransportError(
                f"could not connect/handshake with RL TCP server at {self._host}:{self._port}"
            ) from exc
        except (OSError, ConnectionError) as exc:
            await self._disconnect()
            raise TransportError(
                f"could not connect to RL TCP server at {self._host}:{self._port}"
            ) from exc
        except TransportError:
            await self._disconnect()
            raise

    async def _hello(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        hello = json.dumps(
            {
                "transport_operation": "hello",
                "client_session_id": self._client_session_id,
            },
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        writer.write(hello)
        await writer.drain()
        line = await self._read_response_frame(reader)
        try:
            response = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TransportError("RL hello response is not valid JSON") from exc
        if (
            not isinstance(response, dict)
            or response.get("transport_operation") != "hello"
            or response.get("client_session_id") != self._client_session_id
        ):
            raise TransportError("RL TCP server returned an invalid hello response")
        epoch = response.get("server_epoch")
        if not isinstance(epoch, str) or not epoch:
            raise TransportError("RL hello response is missing server_epoch")
        self._assert_epoch(epoch)

    async def _decode_response(
        self, line: bytes, *, retry_request: RetryRequest
    ) -> JsonObject:
        try:
            response: Any = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            await self._disconnect()
            raise TransportError(
                "RL TCP server returned invalid JSON",
                completion_uncertain=True,
                retry_request=retry_request,
            ) from exc
        if not isinstance(response, dict):
            await self._disconnect()
            raise TransportError(
                "RL TCP response must be a JSON object",
                completion_uncertain=True,
                retry_request=retry_request,
            )
        transport_error = response.get("transport_error")
        if transport_error is not None:
            await self._disconnect()
            detail = response.get("error")
            suffix = f": {detail}" if isinstance(detail, str) and detail else ""
            raise TransportError(
                f"RL TCP transport error {transport_error!r}{suffix}",
                completion_uncertain=False,
            )
        return response

    def _assert_epoch(
        self,
        actual_epoch: str,
        *,
        completion_uncertain: bool = False,
        retry_request: RetryRequest | None = None,
    ) -> None:
        expected = self._server_epoch
        if expected is None:
            self._server_epoch = actual_epoch
            return
        if actual_epoch != expected:
            raise ServerEpochChangedError(
                expected_epoch=expected,
                actual_epoch=actual_epoch,
                completion_uncertain=completion_uncertain,
                retry_request=retry_request,
            )

    async def _read_response_frame(self, reader: asyncio.StreamReader) -> bytes:
        chunks: list[bytes] = []
        total_bytes = 0
        while True:
            remaining = self._max_response_bytes - total_bytes
            if remaining <= 0:
                raise TransportError(
                    f"response exceeds max_response_bytes={self._max_response_bytes}"
                )
            chunk = await reader.read(min(_RESPONSE_READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise RuntimeExitedError(
                    "RL TCP server closed the connection before completing a response frame"
                )
            newline_index = chunk.find(b"\n")
            if newline_index < 0:
                chunks.append(chunk)
                total_bytes += len(chunk)
                continue
            frame_chunk = chunk[: newline_index + 1]
            chunks.append(frame_chunk)
            if newline_index + 1 != len(chunk):
                raise TransportError(
                    "RL TCP server returned data after the response frame delimiter"
                )
            return b"".join(chunks)

    async def _disconnect(self) -> None:
        writer = self._writer
        self._reader = None
        self._writer = None
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass
