from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class RetryRequest:
    """Immutable serialized API request that can be replayed with the same request id.

    The serialized JSON excludes the NDJSON trailing newline. Reconstructing the mapping
    preserves JSON key order, so ``TcpConnection`` will emit the same request bytes when
    the token is retried.
    """

    serialized_payload: str

    @classmethod
    def from_message(cls, message: Mapping[str, Any]) -> "RetryRequest":
        return cls(
            json.dumps(
                dict(message),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    def to_message(self) -> JsonObject:
        value = json.loads(self.serialized_payload)
        if not isinstance(value, dict):
            raise ValueError("retry request must contain a JSON object")
        return value

    @property
    def request_id(self) -> str | None:
        value = self.to_message().get("request_id")
        return value if isinstance(value, str) else None

    @property
    def operation(self) -> str | None:
        value = self.to_message().get("operation")
        return value if isinstance(value, str) else None


class TransportError(RuntimeError):
    """Base exception for failures before a valid API response is received."""

    def __init__(
        self,
        message: str,
        *,
        completion_uncertain: bool = False,
        retry_request: RetryRequest | None = None,
    ) -> None:
        super().__init__(message)
        self.completion_uncertain = completion_uncertain
        self.retry_request = retry_request


class TransportClosedError(TransportError):
    """The transport has already been closed."""


class RuntimeExitedError(TransportError):
    """The owned RL runtime process is no longer alive."""


class RlTransport(Protocol):
    def call(
        self,
        request: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> JsonObject: ...

    def is_alive(self) -> bool: ...

    def close(self) -> None: ...


class FakeTransport:
    """Deterministic in-memory transport used only by unit tests."""

    def __init__(self, responses: list[JsonObject]) -> None:
        self._responses = [dict(response) for response in responses]
        self._alive = True
        self._requests: list[JsonObject] = []
        self._timeouts: list[float] = []

    @property
    def requests(self) -> list[JsonObject]:
        return [dict(request) for request in self._requests]

    @property
    def timeouts(self) -> list[float]:
        return list(self._timeouts)

    def call(
        self,
        request: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> JsonObject:
        if not self._alive:
            raise TransportClosedError("transport is closed")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if not self._responses:
            raise RuntimeError("no prepared response remains")

        self._requests.append(dict(request))
        self._timeouts.append(timeout_s)
        return self._responses.pop(0)

    def close(self) -> None:
        self._alive = False

    def is_alive(self) -> bool:
        return self._alive
