from __future__ import annotations

from collections.abc import Mapping
from typing import Any

JsonObject = dict[str, Any]


class ResponseRouter:
    """Routes future asynchronous responses by private transport internal ID.

    The current `RLApiServerProcess.call()` path is synchronous and does not use this
    class. It is retained for a future Queue-based transport without exposing RL-side
    private queues to Training.
    """

    def __init__(self) -> None:
        self._pending: dict[int, JsonObject] = {}
        self._expired: set[int] = set()

    def expire(self, internal_id: int) -> None:
        self._validate_internal_id(internal_id)
        self._pending.pop(internal_id, None)
        self._expired.add(internal_id)

    def accept(
        self,
        internal_id: int,
        payload: Mapping[str, Any],
    ) -> None:
        self._validate_internal_id(internal_id)
        if internal_id in self._expired:
            # A response that arrives after the caller timed out is discarded once.
            self._expired.remove(internal_id)
            return
        if internal_id in self._pending:
            raise RuntimeError(f"duplicate response for internal_id={internal_id}")
        self._pending[internal_id] = dict(payload)

    def pop(self, internal_id: int) -> JsonObject | None:
        self._validate_internal_id(internal_id)
        if internal_id in self._expired:
            return None
        return self._pending.pop(internal_id, None)

    @staticmethod
    def _validate_internal_id(internal_id: int) -> None:
        if not isinstance(internal_id, int) or isinstance(internal_id, bool):
            raise TypeError("internal_id must be an integer")
        if internal_id <= 0:
            raise ValueError("internal_id must be positive")
