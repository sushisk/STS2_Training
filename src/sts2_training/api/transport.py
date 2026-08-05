from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

JsonObject = dict[str, Any]


class RlTransport(Protocol):
    """Carries JSON-safe requests to an RL Runtime and returns responses."""

    def call(
        self,
        request: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> JsonObject:
        ...

    def is_alive(self) -> bool:
        ...

    def close(self) -> None:
        ...


class FakeTransport:
    """Deterministic test double that records requests and returns queued responses."""

    def __init__(self, responses: list[JsonObject]) -> None:
        self._responses = [dict(response) for response in responses]
        self._requests: list[JsonObject] = []
        self._alive = True

    @property
    def requests(self) -> list[JsonObject]:
        """Return copies so test code cannot mutate the internal history."""

        return [dict(request) for request in self._requests]

    def call(
        self,
        request: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> JsonObject:
        if not self._alive:
            raise RuntimeError("transport is closed")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if not self._responses:
            raise RuntimeError("no responses remain")

        self._requests.append(dict(request))
        return self._responses.pop(0)

    def is_alive(self) -> bool:
        return self._alive

    def close(self) -> None:
        self._alive = False
