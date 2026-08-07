"""Thin asyncio TCP connection for the separately started STS2_RL process."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from sts2_training.api.transport import (
    JsonObject,
    RetryRequest,
    RuntimeExitedError,
    TransportClosedError,
    TransportError,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_MAX_MESSAGE_BYTES = 1024 * 1024
DEFAULT_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_RESPONSE_READ_CHUNK_BYTES = 64 * 1024


class TcpConnection:
    """Persistent newline-delimited JSON connection.

    This class deliberately knows nothing about API operations, instance routing, or
    request/response correlation. It serializes one JSON object, waits for one JSON
    object in reply, and keeps the socket healthy across exchanges.

    ``max_message_bytes`` bounds outbound request frames only, matching the RL TCP
    framing contract. ``max_response_bytes`` is a separate receiver-side safety bound
    so a peer that never sends a newline cannot make response buffering unbounded. The
    response limit intentionally defaults much higher than the request limit.

    Each exchange has one absolute deadline. Waiting for this connection's lock,
    connecting, writing, draining, and reading the response all consume that same
    budget. ``connect_timeout_s`` is an additional upper bound for only the connect
    phase; it never extends the exchange deadline.

    Once a request may have reached RL, timeout, cancellation, response-size overflow,
    or stream/protocol failures mark the result as completion-uncertain, attach an
    immutable ``RetryRequest`` containing the exact JSON payload, and invalidate the
    connection.
    """

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
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
        if max_message_bytes <= 0:
            raise ValueError("max_message_bytes must be positive")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")

        self._host = host
        self._port = port
        self._connect_timeout_s = connect_timeout_s
        self._max_message_bytes = max_message_bytes
        self._max_response_bytes = max_response_bytes
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
        timeout_s: float | None = None,
        deadline: float | None = None,
    ) -> JsonObject:
        deadline = self._resolve_deadline(timeout_s=timeout_s, deadline=deadline)
        retry_request = RetryRequest.from_message(message)
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

                    # From this point onward RL may observe the request even if the
                    # local write/drain or response wait later fails.
                    sent = True
                    self._writer.write(encoded)
                    await self._writer.drain()
                    line = await self._read_response_frame(self._reader)
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
        except RuntimeExitedError as exc:
            if sent:
                await self._disconnect()
            exc.completion_uncertain = sent or exc.completion_uncertain
            if exc.completion_uncertain:
                exc.retry_request = retry_request
            raise
        except TransportError as exc:
            if sent:
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

    @staticmethod
    def _resolve_deadline(
        *,
        timeout_s: float | None,
        deadline: float | None,
    ) -> float:
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
        now = loop.time()
        connect_limit = now + self._connect_timeout_s
        if deadline is None:
            connect_deadline = connect_limit
            limited_by_api_deadline = False
        else:
            connect_deadline = min(deadline, connect_limit)
            limited_by_api_deadline = deadline <= connect_limit

        try:
            async with asyncio.timeout_at(connect_deadline):
                self._reader, self._writer = await asyncio.open_connection(
                    self._host,
                    self._port,
                    limit=self._max_message_bytes + 1,
                )
        except TimeoutError as exc:
            if limited_by_api_deadline:
                raise TransportError(
                    "RL TCP request timed out before send"
                ) from exc
            raise TransportError(
                f"could not connect to RL TCP server at {self._host}:{self._port}"
            ) from exc
        except OSError as exc:
            raise TransportError(
                f"could not connect to RL TCP server at {self._host}:{self._port}"
            ) from exc

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
                if total_bytes >= self._max_response_bytes:
                    raise TransportError(
                        f"response exceeds max_response_bytes={self._max_response_bytes}"
                    )
                continue

            frame_chunk = chunk[: newline_index + 1]
            chunks.append(frame_chunk)
            total_bytes += len(frame_chunk)
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
