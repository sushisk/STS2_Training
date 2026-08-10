from __future__ import annotations

import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from sts2_training.visualizer.log_reader import JsonlLogReader, ReplayLogError
from sts2_training.visualizer.store import EventStore


class Process(Protocol):
    def poll(self) -> int | None: ...


ProcessFactory = Callable[[Sequence[str]], Process]


class LiveRunController:
    """Launch the existing Whole Run CLI and tail its JSONL run event log.

    Run composition intentionally stays in ``sts2_training.runner.start_new_run``.
    The visualizer only starts that entry point, reads the resulting JSONL, and exposes
    lifecycle state to the HTTP layer.
    """

    def __init__(
        self,
        *,
        store: EventStore,
        log_path: str | Path,
        runner_args: Sequence[str] = (),
        process_factory: ProcessFactory | None = None,
    ) -> None:
        self.store = store
        self.log_path = Path(log_path)
        self.runner_args = tuple(_normalize_runner_args(runner_args))
        reserved = {"--run-log", "--selection-log"}.intersection(self.runner_args)
        if reserved:
            option = sorted(reserved)[0]
            raise ValueError(f"runner_args must not include {option}; visualizer owns the live log path")
        self._process_factory = process_factory or self._default_process_factory
        self._reader = JsonlLogReader(self.log_path)
        self._lock = threading.RLock()
        self._process: Process | None = None
        self._state = "idle"
        self._error: str | None = None
        self._result: Any = None

    @staticmethod
    def _default_process_factory(command: Sequence[str]) -> Process:
        return subprocess.Popen(list(command))

    @property
    def command(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "sts2_training.runner.start_new_run",
            *self.runner_args,
            "--run-log",
            str(self.log_path),
        ]

    def start(self) -> bool:
        with self._lock:
            if self._state != "idle":
                return False
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            # Remove stale replay data before handing the path to the runner. The
            # runner's JSONL writer also opens with append=False.
            self.log_path.write_bytes(b"")
            self.store.clear()
            self._reader.reset()
            try:
                self._process = self._process_factory(self.command)
            except BaseException as exc:  # noqa: BLE001 - surface launch failure in UI
                self._state = "failed"
                self._error = f"{type(exc).__name__}: {exc}"
                return True
            self._state = "running"
            return True

    def refresh(self) -> None:
        with self._lock:
            if self._state not in {"running", "completed", "failed"}:
                return
            if self._state == "failed" and self._process is None:
                return

            try:
                for record in self._reader.poll(final=False):
                    self.store.append(record)
            except ReplayLogError as exc:
                self._state = "failed"
                self._error = str(exc)
                return

            if self._process is None:
                return
            exit_code = self._process.poll()
            if exit_code is None:
                return

            # One final read accepts a valid final JSON object even if an external
            # writer omitted the trailing newline.
            try:
                for record in self._reader.poll(final=True):
                    self.store.append(record)
            except ReplayLogError as exc:
                self._state = "failed"
                self._error = str(exc)
                return

            self._result = {"exit_code": exit_code}
            if exit_code == 0:
                self._state = "completed"
            else:
                self._state = "failed"
                self._error = f"runner exited with code {exit_code}"

    def status(self) -> dict[str, Any]:
        self.refresh()
        with self._lock:
            return {
                "state": self._state,
                "error": self._error,
                "result": self._result,
                "log_path": str(self.log_path),
                "event_count": len(self.store),
            }


def _normalize_runner_args(args: Sequence[str]) -> list[str]:
    normalized = list(args)
    if normalized and normalized[0] == "--":
        normalized.pop(0)
    return normalized
