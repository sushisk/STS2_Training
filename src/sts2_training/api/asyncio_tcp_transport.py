"""Asyncio TCP transport for a separately started STS2_RL process."""

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


class AsyncioTcpTransport:
    """Persistent newline-delimited JSON connection to ``API.tcp_server``.

    This is intentionally an async transport rather than an implementation of the
    existing synchronous ``RlTransport`` protocol. It is the minimal foundation for
    Training and RL to run as independently managed OS processes.
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
        self._call_lock = asyncio.Lock()
        self._closed = False

    async def connect(self) -> None:
        if self._closed:
            raise TransportClosedError("transport is closed")
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

    async def ping(self, *, timeout_s: float = 5.0) -> JsonObject:
        response = await self._exchange(
            {"transport_operation": "ping"},
            timeout_s=timeout_s,
        )
        if response.get("transport_operation") != "pong":
            raise TransportError("RL TCP server returned an invalid ping response")
        return response

    async def call(
        self,
        request: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> JsonObject:
        return await self._exchange(dict(request), timeout_s=timeout_s)

    def is_alive(self) -> bool:
        return (
            not self._closed
            and self._reader is not None
            and self._writer is not None
            and not self._reader.at_eof()
            and not self._writer.is_closing()
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        writer = self._writer
        self._reader = None
        self._writer = None
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def __aenter__(self) -> "AsyncioTcpTransport":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _exchange(
        self,
        request: JsonObject,
        *,
        timeout_s: float,
    ) -> JsonObject:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self._closed:
            raise TransportClosedError("transport is closed")
        if not self.is_alive():
            await self.connect()

        assert self._reader is not None
        assert self._writer is not None
        encoded = json.dumps(
            request,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        if len(encoded) > self._max_message_bytes:
            raise TransportError(
                f"request exceeds max_message_bytes={self._max_message_bytes}"
            )

        async with self._call_lock:
            try:
                response = await asyncio.wait_for(
                    self._round_trip(encoded),
                    timeout=timeout_s,
                )
            except TimeoutError as exc:
                raise TransportError("RL TCP request timed out") from exc
            except (ConnectionError, OSError) as exc:
                raise RuntimeExitedError(
                    "RL TCP connection closed during request"
                ) from exc

        return response

    async def _round_trip(self, encoded: bytes) -> JsonObject:
        assert self._reader is not None
        assert self._writer is not None
        self._writer.write(encoded)
        await self._writer.drain()
        try:
            line = await self._reader.readline()
        except (ValueError, asyncio.LimitOverrunError) as exc:
            raise TransportError("RL TCP response exceeds max_message_bytes") from exc
        if not line:
            raise RuntimeExitedError("RL TCP server closed the connection")
        if len(line) > self._max_message_bytes:
            raise TransportError("RL TCP response exceeds max_message_bytes")
        try:
            response: Any = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TransportError("RL TCP server returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise TransportError("RL TCP response must be a JSON object")
        return response
