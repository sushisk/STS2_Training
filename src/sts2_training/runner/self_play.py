"""Self-play driver (data-collection track): runs many independent Whole Runs
concurrently against one RL server, each through `start_new_run`, and logs every
selection Training makes - the committed action AND every beam-search-explored-
but-unchosen branch - to its own JSONL file via `selection_log.JsonlSelectionLogger`.

This does not add any new decision logic. The policy driving these runs is still
`CombatDecisionEngine`'s heuristic default (`PriorHeuristicPolicy` + beam search for
card/potion/system, `HeuristicCombatSelector` - explicitly built as placeholder
"initial data-collection stage" logic - for everything else, e.g. map/shop/rest/
reward choices). The point of this module is only to drive many such runs at once
and capture what happened, as bootstrap training data (Run-level Win/Lose labels)
for a future board/deck evaluation model.

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
import sys
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sts2_training.api import AsyncTrainingApiClient, TcpConnection
from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.runner._cli import add_common_arguments, configure_logging
from sts2_training.runner.episode import EpisodeResult
from sts2_training.runner.start_new_run import start_new_run
from sts2_training.selection_log import JsonlSelectionLogger

__all__ = ["SelfPlayRunResult", "run_self_play_batch"]

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelfPlayRunResult:
    """One `start_new_run` attempt. `episode` is `None` iff `error` is set - a
    failed run still gets a `log_path` (whatever was recorded before it failed)."""

    run_id: str
    log_path: Path
    episode: EpisodeResult | None
    error: str | None


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


async def _run_one(
    run_id: str,
    *,
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
    logger = JsonlSelectionLogger(log_path)
    try:
        connection = connection_factory()
        async with AsyncTrainingApiClient(connection, selection_logger=logger) as client:
            episode = await start_new_run(
                client,
                character_id=character_id,
                ascension=ascension,
                decision_timeout_s=decision_timeout_s,
                max_decisions=max_decisions,
                search_mode=search_mode,
                beam_max_depth=beam_max_depth,
            )
        return SelfPlayRunResult(run_id, log_path, episode, None)
    except Exception as exc:  # noqa: BLE001 - one run's failure must not sink the batch
        _LOG.exception("self-play run %s failed", run_id)
        return SelfPlayRunResult(run_id, log_path, None, f"{type(exc).__name__}: {exc}")
    finally:
        logger.close()


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
    """Run `num_runs` independent `start_new_run` episodes, at most `concurrency` at
    once. Each run gets its own connection (default: a fresh `TcpConnection(host,
    port, connect_timeout_s)` per run - pass `connection_factory` to point at
    something else, e.g. a fake in tests) and its own JSONL log file under
    `output_dir`, named `<character_id>-<batch_start_time>-<index>-<random>.jsonl`.

    One run failing (start rejected, transport error, emulator fault, ...) is
    captured in that run's `SelfPlayRunResult.error` rather than raised - a bad
    server-side interaction on run 7 of 50 should not discard the other 49.
    """
    if isinstance(num_runs, bool) or not isinstance(num_runs, int) or num_runs <= 0:
        raise ValueError("num_runs must be a positive integer")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
        raise ValueError("concurrency must be a positive integer")

    factory = connection_factory or (
        lambda: TcpConnection(host=host, port=port, connect_timeout_s=connect_timeout_s)
    )
    semaphore = asyncio.Semaphore(concurrency)
    batch_tag = f"{character_id.lower()}-{int(time.time())}"

    async def _bounded(index: int) -> SelfPlayRunResult:
        run_id = f"{batch_tag}-{index:05d}-{uuid.uuid4().hex[:8]}"
        async with semaphore:
            return await _run_one(
                run_id,
                connection_factory=factory,
                character_id=character_id,
                ascension=ascension,
                decision_timeout_s=decision_timeout_s,
                max_decisions=max_decisions,
                search_mode=search_mode,
                beam_max_depth=beam_max_depth,
                output_dir=output_dir,
            )

    return await asyncio.gather(*[_bounded(index) for index in range(num_runs)])


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
        "failures": [{"run_id": r.run_id, "error": r.error} for r in failed],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
