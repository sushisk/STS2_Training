from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any


class EventStore:
    """Thread-safe in-memory store of records already read from JSONL."""

    def __init__(self, records: Iterable[Mapping[str, Any]] = ()) -> None:
        self._lock = threading.RLock()
        self._records: list[dict[str, Any]] = [deepcopy(dict(record)) for record in records]

    def append(self, record: Mapping[str, Any]) -> int:
        with self._lock:
            self._records.append(deepcopy(dict(record)))
            return len(self._records) - 1

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def after(self, cursor: int) -> list[tuple[int, dict[str, Any]]]:
        """Return records whose zero-based index is greater than ``cursor``."""
        with self._lock:
            start = max(cursor + 1, 0)
            return [
                (index, deepcopy(record))
                for index, record in enumerate(self._records[start:], start)
            ]

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


__all__ = ["EventStore"]
