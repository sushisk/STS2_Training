"""Self-play driver (data-collection track): runs many independent Whole Runs
concurrently against one RL server, each through `start_new_run`, and logs every
selection Training makes - the committed action AND every beam-search-explored-
but-unchosen branch - to its own JSONL file via `selection_log.JsonlSelectionLogger`.
A successfully and completely logged run appends one `self_play_run_result` record
with the run seed/configuration and terminal outcome so each file carries its own
Run-level Win/Lose label.

This does not add any new decision logic. The policy driving these runs is still
`CombatDecisionEngine`'s heuristic default (`PriorHeuristicPolicy` + beam search for
card/potion/system, `HeuristicCombatSelector` - explicitly built as placeholder
"initial data-collection stage" logic - for everything else, e.g. map/shop/rest/
reward choices). The point of this module is only to drive many such runs at once
and capture what happened, as bootstrap training data (Run-level Win/Lose labels)
for a future board/deck evaluation model. Each run's generated game seed is embedded
in its run ID / JSONL filename and also seeds the random fallback selector, so the
game setup and fallback-policy choices can be replayed deterministically.

CLI use::

    python -m sts2_training.runner.self_play \\
        --host 127.0.0.1 --port 8765 --character-id IRONCLAD \\
        --num-runs 50 --concurrency 8 --output-dir data/self_play
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
import re
import sys
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sts2_training.api import AsyncTrainingApiClient, TcpConnection
from sts2_training.decision import CombatDecisionEngine
from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.decision.search_modes import resolve_search_mode
from sts2_training.runner._cli import _positive_int, add_common_arguments, configure_logging
from sts2_training.runner.episode import EpisodeResult
from sts2_training.runner.start_new_run import start_new_run
from sts2_training.selection.heuristic_selector import HeuristicCombatSelector
from sts2_training.selection_log import JsonlSelectionLogger

__all__ = ["SelfPlayRunResult", "run_self_play_batch"]

_LOG = logging.getLogger(__name__)
_FILENAME_UNSAFE = re.compile(r"[^a-z0-9._-]+")
_MAX_GAME_SEED = 2**31 - 1


@dataclass(frozen=True)
class SelfPlayRunResult:
    """One `start_new_run` attempt.

    `episode` is `None` iff `error` is set. A failed run still gets a `log_path`;
    when logging itself fails, the path may contain only the records flushed before
    the failure (or may not exist if logger creation failed).
    """

    run_id: str
    log_path: Path
    episode: EpisodeResult | None
    error: str | None


def _format_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_optional_positive_int(name: str, value: int | None) -> None:
    if value is not None:
        _validate_positive_int(name, value)


def _validate_positive_float(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")


def _filename_component(value: str) -> str:
    """Return a path-safe, human-readable component for a run filename."""
    normalized = _FILENAME_UNSAFE.sub("-", value.strip().lower()).strip("._-")
    return normalized or "character"


def _seed_for_index(batch_seed: int, index: int) -> int:
    """Map a batch seed plus zero-based run index to a unique valid game seed."""
    return ((batch_seed - 1 + index) % _MAX_GAME_SEED) + 1


def _run_result_event(
    *,
    run_id: str,
    seed: int,
    character_id: str,
    ascension: int,
    episode: EpisodeResult,
) -> dict[str, Any]:
    """Build the terminal per-run record that labels a complete self-play log."""
    return {
        "event": "self_play_run_result",
        "run_id": run_id,
        "seed": seed,
        "character_id": character_id,
        "ascension": ascension,
        "instance_id": episode.instance_id,
        "decisions_made": episode.decisions_made,
        "elapsed_s": episode.elapsed_s,
        "outcome": episode.final_dto.get("outcome"),
        "final_dto": episode.final_dto,
    }


class _TrackingSelectionLogger:
    """Remember logger write failures that SelectionAudit intentionally swallows.

    Selection logging is globally best-effort so a generic client never changes game
    behavior when audit output fails. Self-play is different: its product *is* the
    data, so the run may finish playing but must still be reported as failed when the
    JSONL stream became incomplete.
    """

    def __init__(self, logger: JsonlSelectionLogger) -> None:
        self._logger = logger
        self.error: BaseException | None = None

    def __call__(self, event: Mapping[str, Any]) -> None:
        try:
            self._logger(event)
        except Exception as exc:
            if self.error is None:
                self.error = exc
            raise


async def _close_client_best_effort(client: AsyncTrainingApiClient, run_id: str) -> None:
    try:
        await client.close()
    except Exception:  # noqa: BLE001 - transport cleanup must not overwrite run result
        _LOG.exception("self-play run %s: client close failed", run_id)


async def _run_one(
    run_id: str,
    *,
    seed: int,
    connection_factory: Callable[[], Any],
    character_id: str,
    ascension: int,
    decision_timeout_s: float,
    max_decisions: int | None,
    search_mode: str | BeamSearchConfig | None,
    beam_max_depth: int | None,
    output_dir: Path,
) -> SelfPlayRunResult:
    log_path = output_dir / f"{run_id}.jsonl"
    logger: JsonlSelectionLogger | None = None
    tracked_logger: _TrackingSelectionLogger | None = None
    client: AsyncTrainingApiClient | None = None
    episode: EpisodeResult | None = None
    error: str | None = None

    try:
        logger = JsonlSelectionLogger(log_path)
        tracked_logger = _TrackingSelectionLogger(logger)
        connection = connection_factory()
        client = AsyncTrainingApiClient(connection, selection_logger=tracked_logger)
        await connection.connect()
        engine = CombatDecisionEngine(
            client,
            beam_config=resolve_search_mode(search_mode, max_depth=beam_max_depth),
            fallback_selector=HeuristicCombatSelector(random.Random(seed)),
        )
        episode = await start_new_run(
            client,
            character_id=character_id,
            ascension=ascension,
            seed=seed,
            decision_timeout_s=decision_timeout_s,
            max_decisions=max_decisions,
            engine=engine,
        )
        # SelectionAudit deliberately swallows logger failures so gameplay can continue.
        # Do not append a terminal "complete" record to a log already known to have a
        # hole. A failure while writing this final record is tracked and reported below.
        if tracked_logger.error is None:
            try:
                tracked_logger(
                    _run_result_event(
                        run_id=run_id,
                        seed=seed,
                        character_id=character_id,
                        ascension=ascension,
                        episode=episode,
                    )
                )
            except Exception:  # noqa: BLE001 - tracked logger owns failure reporting
                pass
    except Exception as exc:  # noqa: BLE001 - one run's failure must not sink the batch
        _LOG.exception("self-play run %s failed", run_id)
        episode = None
        error = _format_error(exc)
    finally:
        try:
            if client is not None:
                await _close_client_best_effort(client, run_id)
        finally:
            # Even cancellation during transport cleanup must not leave the JSONL
            # stream open or skip its final flush/close attempt.
            if logger is not None:
                try:
                    logger.close()
                except Exception as exc:  # noqa: BLE001 - contain the failure to this run
                    _LOG.exception("self-play run %s: selection log close failed", run_id)
                    if error is None:
                        episode = None
                        error = f"SelectionLogCloseError: {_format_error(exc)}"

    if tracked_logger is not None and tracked_logger.error is not None and error is None:
        episode = None
        error = f"SelectionLogError: {_format_error(tracked_logger.error)}"

    return SelfPlayRunResult(run_id, log_path, episode, error)


async def run_self_play_batch(
    *,
    character_id: str,
    num_runs: int,
    concurrency: int = 4,
    ascension: int = 0,
    connection_factory: Callable[[], Any] | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    connect_timeout_s: float = 5.0,
    decision_timeout_s: float = 30.0,
    max_decisions: int | None = None,
    search_mode: str | BeamSearchConfig | None = None,
    beam_max_depth: int | None = None,
    output_dir: Path = Path("data/self_play"),
) -> list[SelfPlayRunResult]:
    """Run `num_runs` independent `start_new_run` episodes with bounded workers.

    Each run gets its own connection (default: a fresh `TcpConnection(host, port,
    connect_timeout_s)`) and JSONL file under `output_dir`. Shared configuration is
    validated before any connection is opened. Runtime and logging failures are
    captured per run so one bad interaction does not discard the rest of the batch.
    One random batch seed is expanded to unique per-run game seeds, each embedded in
    its run ID and reused to seed the fallback selector for deterministic replay.
    """
    _validate_positive_int("num_runs", num_runs)
    if num_runs > _MAX_GAME_SEED:
        raise ValueError(f"num_runs must be <= {_MAX_GAME_SEED} to keep game seeds unique")
    _validate_positive_int("concurrency", concurrency)
    if not isinstance(character_id, str) or not character_id.strip():
        raise ValueError("character_id must be a non-empty string")
    if isinstance(ascension, bool) or not isinstance(ascension, int):
        raise ValueError("ascension must be an integer")
    _validate_positive_float("decision_timeout_s", decision_timeout_s)
    _validate_optional_positive_int("max_decisions", max_decisions)
    # Validate the shared beam configuration once, before spawning any network work.
    resolve_search_mode(search_mode, max_depth=beam_max_depth)

    if connection_factory is None:
        _validate_positive_float("connect_timeout_s", connect_timeout_s)
        # Construct once only for validation; each run still receives a fresh connection.
        TcpConnection(host=host, port=port, connect_timeout_s=connect_timeout_s)
        factory: Callable[[], Any] = lambda: TcpConnection(
            host=host, port=port, connect_timeout_s=connect_timeout_s
        )
    else:
        factory = connection_factory

    output_dir = Path(output_dir)
    batch_tag = f"{_filename_component(character_id)}-{int(time.time())}"
    batch_seed = random.randint(1, _MAX_GAME_SEED)
    results: list[SelfPlayRunResult | None] = [None] * num_runs
    next_index = 0
    index_lock = asyncio.Lock()

    async def _worker() -> None:
        nonlocal next_index
        while True:
            async with index_lock:
                if next_index >= num_runs:
                    return
                index = next_index
                next_index += 1

            seed = _seed_for_index(batch_seed, index)
            run_id = f"{batch_tag}-{index:05d}-seed-{seed}-{uuid.uuid4().hex[:8]}"
            results[index] = await _run_one(
                run_id,
                seed=seed,
                connection_factory=factory,
                character_id=character_id,
                ascension=ascension,
                decision_timeout_s=decision_timeout_s,
                max_decisions=max_decisions,
                search_mode=search_mode,
                beam_max_depth=beam_max_depth,
                output_dir=output_dir,
            )

    workers = [asyncio.create_task(_worker()) for _ in range(min(concurrency, num_runs))]
    await asyncio.gather(*workers)
    return [result for result in results if result is not None]


def _summarize(results: list[SelfPlayRunResult]) -> dict[str, Any]:
    completed = [r for r in results if r.episode is not None]
    failed = [r for r in results if r.episode is None]
    outcome_counts = Counter(r.episode.final_dto.get("outcome") for r in completed)
    return {
        "runs_requested": len(results),
        "runs_completed": len(completed),
        "runs_failed": len(failed),
        "outcome_counts": dict(outcome_counts),
        "avg_decisions_made": (
            sum(r.episode.decisions_made for r in completed) / len(completed) if completed else None
        ),
        "avg_elapsed_s": (
            sum(r.episode.elapsed_s for r in completed) / len(completed) if completed else None
        ),
        "failures": [
            {"run_id": r.run_id, "log_path": str(r.log_path), "error": r.error} for r in failed
        ],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser)
    parser.add_argument("--character-id", required=True)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--num-runs", type=_positive_int, required=True)
    parser.add_argument("--concurrency", type=_positive_int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("data/self_play"))
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> list[SelfPlayRunResult]:
    return await run_self_play_batch(
        character_id=args.character_id,
        ascension=args.ascension,
        num_runs=args.num_runs,
        concurrency=args.concurrency,
        host=args.host,
        port=args.port,
        connect_timeout_s=args.connect_timeout,
        decision_timeout_s=args.decision_timeout,
        max_decisions=args.max_decisions,
        search_mode=args.search_mode,
        beam_max_depth=args.beam_depth,
        output_dir=args.output_dir,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging(args.log_level)
    results = asyncio.run(_run(args))
    summary = _summarize(results)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 1 if summary["runs_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
