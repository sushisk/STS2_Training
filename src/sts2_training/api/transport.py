from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

JsonObject = dict[str, Any]


class TransportError(RuntimeError):
    """Base exception for failures before a valid API response is received."""


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
