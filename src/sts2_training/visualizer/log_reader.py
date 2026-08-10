from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ReplayLogError(ValueError):
    """Raised when a JSONL visualizer input cannot be decoded safely."""


class JsonlLogReader:
    """Incrementally read complete JSONL records from a growing file.

    The reader tracks a byte offset and retains an unterminated tail between polls.
    This makes the same reader suitable for completed replay logs and live logs that
    may be observed while their final JSON object is still being written.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._offset = 0
        self._pending = b""
        self._line_number = 0

    def poll(self, *, final: bool = False) -> list[dict[str, Any]]:
        try:
            with self.path.open("rb") as stream:
                stream.seek(0, 2)
                size = stream.tell()
                if size < self._offset:
                    raise ReplayLogError(f"visualizer log was truncated while reading {self.path}")
                stream.seek(self._offset)
                chunk = stream.read()
        except FileNotFoundError:
            if final:
                raise ReplayLogError(f"cannot open visualizer log {self.path}: file does not exist")
            return []
        except OSError as exc:
            raise ReplayLogError(f"cannot open visualizer log {self.path}: {exc}") from exc

        self._offset += len(chunk)
        data = self._pending + chunk
        parts = data.split(b"\n")
        self._pending = parts.pop()

        records: list[dict[str, Any]] = []
        for raw_line in parts:
            self._line_number += 1
            record = self._decode_line(raw_line, self._line_number)
            if record is not None:
                records.append(record)

        if final and self._pending:
            self._line_number += 1
            record = self._decode_line(self._pending, self._line_number)
            self._pending = b""
            if record is not None:
                records.append(record)
        return records

    def _decode_line(self, raw_line: bytes, line_number: int) -> dict[str, Any] | None:
        if not raw_line.strip():
            return None
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReplayLogError(
                f"invalid UTF-8 in {self.path} at line {line_number}"
            ) from exc
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayLogError(
                f"invalid JSON in {self.path} at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise ReplayLogError(
                f"record in {self.path} at line {line_number} must be a JSON object"
            )
        return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a completed JSONL log through the same reader used by live mode."""
    return JsonlLogReader(path).poll(final=True)


__all__ = ["JsonlLogReader", "ReplayLogError", "read_jsonl"]
