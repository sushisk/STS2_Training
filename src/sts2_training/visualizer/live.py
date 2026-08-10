from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from sts2_training.visualizer.core import EventStore


class EventWriter(Protocol):
    def __call__(self, event: Mapping[str, Any]) -> None: ...
    def close(self) -> None: ...


Runner = Callable[[Callable[[Mapping[str, Any]], None]], Awaitable[Any]]
WriterFactory = Callable[[Path], EventWriter]


@dataclass(frozen=True)
class LiveRunConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    connect_timeout_s: float = 5.0
    character_id: str = "IRONCLAD"
    ascension: int = 0
    seed: int | None = None
    decision_timeout_s: float = 30.0
    max_decisions: int | None = None
    search_mode: str | None = None
    beam_max_depth: int | None = None


class _FanoutLogger:
    def __init__(self, store: EventStore, writer: EventWriter) -> None:
        self._store = store
        self._writer = writer

    def __call__(self, event: Mapping[str, Any]) -> None:
        # JsonlSelectionLogger adds logged_at to its private copy. Stamp the fanout
        # record first so the live browser and persisted replay see the same timestamp.
        record = dict(event)
        record.setdefault(
            "logged_at",
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        # Keep the live display useful even when persistence fails. SelectionAudit
        # already treats logger failures as best-effort and must not alter gameplay.
        self._store.append(record)
        self._writer(record)


class LiveRunController:
    """Own one visualized Whole Run and expose thread-safe lifecycle status."""

    def __init__(
        self,
        *,
        store: EventStore,
        config: LiveRunConfig,
        log_path: str | Path,
        runner: Runner | None = None,
        writer_factory: WriterFactory | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self.log_path = Path(log_path)
        self._runner = runner or self._default_runner
        self._writer_factory = writer_factory or self._default_writer_factory
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._state = "idle"
        self._error: str | None = None
        self._result: Any = None

    @staticmethod
    def _default_writer_factory(path: Path) -> EventWriter:
        from sts2_training.selection_log import JsonlSelectionLogger

        return JsonlSelectionLogger(path, append=False)

    async def _default_runner(self, event_logger: Callable[[Mapping[str, Any]], None]) -> Any:
        from sts2_training.api import AsyncTrainingApiClient, TcpConnection
        from sts2_training.runner.start_new_run import start_new_run

        connection = TcpConnection(
            host=self.config.host,
            port=self.config.port,
            connect_timeout_s=self.config.connect_timeout_s,
        )
        async with AsyncTrainingApiClient(connection, selection_logger=event_logger) as client:
            return await start_new_run(
                client,
                character_id=self.config.character_id,
                ascension=self.config.ascension,
                seed=self.config.seed,
                decision_timeout_s=self.config.decision_timeout_s,
                max_decisions=self.config.max_decisions,
                search_mode=self.config.search_mode,
                beam_max_depth=self.config.beam_max_depth,
            )

    def start(self) -> bool:
        with self._lock:
            if self._state != "idle":
                return False
            self._state = "running"
            self._thread = threading.Thread(
                target=self._thread_main,
                name="sts2-visualizer-live-run",
                daemon=True,
            )
            self._thread.start()
            return True

    def _thread_main(self) -> None:
        writer: EventWriter | None = None
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            writer = self._writer_factory(self.log_path)
            result = asyncio.run(self._runner(_FanoutLogger(self.store, writer)))
        except BaseException as exc:  # noqa: BLE001 - surface runner failure in the UI
            with self._lock:
                self._state = "failed"
                self._error = f"{type(exc).__name__}: {exc}"
            return
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception as exc:  # noqa: BLE001
                    with self._lock:
                        if self._error is None:
                            self._state = "failed"
                            self._error = f"LogCloseError: {type(exc).__name__}: {exc}"

        with self._lock:
            if self._state != "failed":
                self._state = "completed"
                self._result = _result_summary(result)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "error": self._error,
                "result": self._result,
                "log_path": str(self.log_path),
                "event_count": len(self.store),
            }

    def join(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)


def _result_summary(result: Any) -> Any:
    if isinstance(result, Mapping):
        return dict(result)
    fields = {}
    for name in ("instance_id", "decisions_made", "elapsed_s", "final_dto"):
        if hasattr(result, name):
            fields[name] = getattr(result, name)
    return fields or repr(result)
