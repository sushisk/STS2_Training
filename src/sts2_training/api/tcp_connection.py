"""Thin asyncio TCP connection for the separately started STS2_RL process."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from sts2_training.api.transport import (
    JsonObject,
    RuntimeExitedError,
    TransportClosedError,
    TransportError,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_MAX_MESSAGE_BYTES = 1024 * 1024
_RESPONSE_READ_CHUNK_BYTES = 64 * 1024


class TcpConnection:
    """Persistent newline-delimited JSON connection.

    This class deliberately knows nothing about API operations, instance routing, or
    request/response correlation. It only serializes one JSON object, waits for one JSON
    object in reply, and keeps the socket healthy across exchanges.

    ``max_message_bytes`` limits outbound request frames only. Responses are read until
    their newline without applying that request limit. This is important for ambiguous
    completion: a completed non-idempotent API operation must not become permanently
    unrecoverable merely because its response is larger than the request-frame limit.

    Once a request has started writing, cancellation/timeout/stream errors invalidate the
    connection. A later exchange therefore reconnects instead of consuming a late response
    from the cancelled request.
    """

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        connect_timeout_s: float = 5.0,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        if not host:
            raise ValueError("host must not be empty")
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be positive")
        if max_message_bytes <= 0:
            raise ValueError("max_message_bytes must be positive")

        self._host = host
        self._port = port
        self._connect_timeout_s = connect_timeout_s
        self._max_message_bytes = max_message_bytes
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def connect(self) -> None:
        async with self._lock:
            await self._connect()

    async def exchange(
        self,
        message: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> JsonObject:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        encoded = json.dumps(
            dict(message),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        if len(encoded) > self._max_message_bytes:
            raise TransportError(
                f"request exceeds max_message_bytes={self._max_message_bytes}"
            )

        # The v0.5 TCP contract is intentionally one request -> one response per
        # connection at a time. No transport-level correlation ID or response router
        # is needed; API request_id remains an API-level concern.
        async with self._lock:
            await self._connect()
            assert self._reader is not None
            assert self._writer is not None
            sent = False
            try:
                async with asyncio.timeout(timeout_s):
                    sent = True
                    self._writer.write(encoded)
                    await self._writer.drain()
                    line = await self._read_response_frame(self._reader)
            except asyncio.CancelledError:
                if sent:
                    await self._disconnect()
                raise
            except TimeoutError as exc:
                await self._disconnect()
                raise TransportError("RL TCP request timed out") from exc
            except (ConnectionError, OSError) as exc:
                await self._disconnect()
                raise RuntimeExitedError(
                    "RL TCP connection closed during request"
                ) from exc
            except TransportError:
                await self._disconnect()
                raise

            if not line:
                await self._disconnect()
                raise RuntimeExitedError("RL TCP server closed the connection")
            try:
                response: Any = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                await self._disconnect()
                raise TransportError("RL TCP server returned invalid JSON") from exc
            if not isinstance(response, dict):
                await self._disconnect()
                raise TransportError("RL TCP response must be a JSON object")

            transport_error = response.get("transport_error")
            if transport_error is not None:
                await self._disconnect()
                detail = response.get("error")
                suffix = f": {detail}" if isinstance(detail, str) and detail else ""
                raise TransportError(
                    f"RL TCP transport error {transport_error!r}{suffix}"
                )
            return response

    async def ping(self, *, timeout_s: float = 5.0) -> JsonObject:
        response = await self.exchange(
            {"transport_operation": "ping"},
            timeout_s=timeout_s,
        )
        if response != {"transport_operation": "pong"}:
            await self.invalidate()
            raise TransportError("RL TCP server returned an invalid ping response")
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
        """Discard the current stream without permanently closing this connection."""
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

    async def _connect(self) -> None:
        if self._closed:
            raise TransportClosedError("connection is closed")
        if self.is_alive():
            return
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(
                    self._host,
                    self._port,
                    limit=self._max_message_bytes + 1,
                ),
                timeout=self._connect_timeout_s,
            )
        except (TimeoutError, OSError) as exc:
            raise TransportError(
                f"could not connect to RL TCP server at {self._host}:{self._port}"
            ) from exc

    @staticmethod
    async def _read_response_frame(reader: asyncio.StreamReader) -> bytes:
        chunks: list[bytes] = []
        while True:
            chunk = await reader.read(_RESPONSE_READ_CHUNK_BYTES)
            if not chunk:
                if chunks:
                    raise TransportError(
                        "RL TCP server closed the connection before response newline"
                    )
                return b""

            newline_index = chunk.find(b"\n")
            if newline_index < 0:
                chunks.append(chunk)
                continue

            chunks.append(chunk[: newline_index + 1])
            if chunk[newline_index + 1 :]:
                # With one outstanding request per connection, bytes after the first
                # response newline mean the peer violated the v0.5 framing contract.
                raise TransportError(
                    "RL TCP server returned multiple response frames for one request"
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
