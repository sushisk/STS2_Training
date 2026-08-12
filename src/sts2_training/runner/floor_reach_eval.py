"""Evaluate floor reach across independently seeded Whole Runs.

Matched-seed before/after comparisons are not meaningful for policy changes because a
single changed decision alters the rest of the trajectory. Compare aggregate statistics
across independent runs instead.

CLI use::

    python -m sts2_training.runner.floor_reach_eval --character-id IRONCLAD --num-runs 30
    python -m sts2_training.runner.floor_reach_eval --character-id IRONCLAD --num-runs 30 --no-beam
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import logging
import random
import statistics
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sts2_training.api import AsyncTrainingApiClient, TcpConnection
from sts2_training.decision import CombatDecisionEngine
from sts2_training.decision import engine as engine_module
from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.decision.search_modes import resolve_search_mode
from sts2_training.runner._cli import _positive_int, add_common_arguments, configure_logging
from sts2_training.runner.start_new_run import start_new_run
from sts2_training.selection.heuristic_selector import HeuristicCombatSelector

__all__ = [
    "FloorReachResult",
    "run_floor_reach_eval",
    "summarize_floor_reach",
]

_LOG = logging.getLogger(__name__)
_MAX_GAME_SEED = 2**31 - 1


@dataclass(frozen=True)
class FloorReachResult:
    """Result of one run, including the deepest observed ``totalFloor``."""

    run_id: str
    seed: int
    max_total_floor: int
    act_index_at_max: int | None
    decisions_made: int
    decision_source_counts: dict[str, int]
    outcome: str | None
    error: str | None
    elapsed_s: float


@dataclass
class _RunState:
    max_total_floor: int = 0
    act_index_at_max: int | None = None
    decisions_made: int = 0
    source_counts: Counter[str] = field(default_factory=Counter)


_current_state: contextvars.ContextVar[_RunState | None] = contextvars.ContextVar(
    "_floor_reach_eval_current_state", default=None
)
_orig_decide = engine_module.CombatDecisionEngine.decide
_patched = False


def _record_floor(state: _RunState, dto: Mapping[str, Any]) -> None:
    total_floor = dto.get("totalFloor")
    if not isinstance(total_floor, int) or total_floor <= state.max_total_floor:
        return

    state.max_total_floor = total_floor
    act_index = dto.get("currentActIndex")
    state.act_index_at_max = act_index if isinstance(act_index, int) else None


async def _tracking_decide(self: CombatDecisionEngine, *args: Any, **kwargs: Any) -> Any:
    outcome = await _orig_decide(self, *args, **kwargs)
    state = _current_state.get()
    if state is None:
        return outcome

    state.decisions_made += 1
    state.source_counts[outcome.source] += 1
    decision = outcome.decision
    if isinstance(decision, Mapping):
        dto = decision.get("masked_emulator_dto")
        if isinstance(dto, Mapping):
            _record_floor(state, dto)
    return outcome


def _ensure_patched() -> None:
    """Install the tracking wrapper once; ``ContextVar`` keeps task state isolated."""
    global _patched
    if not _patched:
        engine_module.CombatDecisionEngine.decide = _tracking_decide
        _patched = True


def _build_engine(
    client: Any,
    *,
    seed: int,
    use_beam: bool,
    search_mode: str | BeamSearchConfig | None,
    beam_max_depth: int | None,
) -> CombatDecisionEngine:
    fallback_selector = HeuristicCombatSelector(random.Random(seed))
    if use_beam:
        beam_config = resolve_search_mode(search_mode, max_depth=beam_max_depth)
        return CombatDecisionEngine(
            client, beam_config=beam_config, fallback_selector=fallback_selector
        )
    return CombatDecisionEngine(
        client, beam_action_types=frozenset(), fallback_selector=fallback_selector
    )


async def _run_one(
    run_id: str,
    *,
    seed: int,
    connection_factory: Callable[[], Any],
    character_id: str,
    ascension: int,
    decision_timeout_s: float,
    max_decisions: int | None,
    use_beam: bool,
    search_mode: str | BeamSearchConfig | None,
    beam_max_depth: int | None,
) -> FloorReachResult:
    _ensure_patched()
    state = _RunState()
    token = _current_state.set(state)
    t0 = time.monotonic()
    outcome_label: str | None = None
    error: str | None = None
    client: AsyncTrainingApiClient | None = None
    try:
        connection = connection_factory()
        client = AsyncTrainingApiClient(connection)
        await connection.connect()
        engine = _build_engine(
            client,
            seed=seed,
            use_beam=use_beam,
            search_mode=search_mode,
            beam_max_depth=beam_max_depth,
        )
        result = await start_new_run(
            client,
            character_id=character_id,
            ascension=ascension,
            seed=seed,
            decision_timeout_s=decision_timeout_s,
            max_decisions=max_decisions,
            engine=engine,
        )
        outcome_label = result.final_dto.get("outcome")
        _record_floor(state, result.final_dto)
    except Exception as exc:  # noqa: BLE001 - preserve the rest of the batch
        _LOG.exception("floor-reach-eval run %s failed", run_id)
        error = f"{type(exc).__name__}: {exc}"
    finally:
        _current_state.reset(token)
        if client is not None:
            try:
                await client.close()
            except Exception:  # noqa: BLE001 - cleanup must not replace the run result
                _LOG.exception("floor-reach-eval run %s: client close failed", run_id)

    return FloorReachResult(
        run_id=run_id,
        seed=seed,
        max_total_floor=state.max_total_floor,
        act_index_at_max=state.act_index_at_max,
        decisions_made=state.decisions_made,
        decision_source_counts=dict(state.source_counts),
        outcome=outcome_label,
        error=error,
        elapsed_s=round(time.monotonic() - t0, 1),
    )


async def run_floor_reach_eval(
    *,
    character_id: str,
    num_runs: int,
    concurrency: int = 1,
    ascension: int = 0,
    use_beam: bool = True,
    connection_factory: Callable[[], Any] | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    connect_timeout_s: float = 5.0,
    decision_timeout_s: float = 90.0,
    max_decisions: int | None = 600,
    search_mode: str | BeamSearchConfig | None = None,
    beam_max_depth: int | None = None,
) -> list[FloorReachResult]:
    """Run independent episodes and track the deepest floor reached by each.

    ``concurrency`` defaults to 1 because the plain TCP server uses one shared Emulator
    ``GameInstance``. Raise it only when the server supports independent instances.
    """
    if not isinstance(num_runs, int) or isinstance(num_runs, bool) or num_runs <= 0:
        raise ValueError("num_runs must be a positive integer")
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency <= 0:
        raise ValueError("concurrency must be a positive integer")
    if not isinstance(character_id, str) or not character_id.strip():
        raise ValueError("character_id must be a non-empty string")
    resolve_search_mode(search_mode, max_depth=beam_max_depth)

    if connection_factory is None:
        factory: Callable[[], Any] = lambda: TcpConnection(
            host=host, port=port, connect_timeout_s=connect_timeout_s
        )
    else:
        factory = connection_factory

    batch_tag = f"{character_id.lower()}-{int(time.time())}"
    results: list[FloorReachResult | None] = [None] * num_runs
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

            seed = random.randint(1, _MAX_GAME_SEED)
            run_id = f"{batch_tag}-{index:05d}-seed-{seed}-{uuid.uuid4().hex[:8]}"
            results[index] = await _run_one(
                run_id,
                seed=seed,
                connection_factory=factory,
                character_id=character_id,
                ascension=ascension,
                decision_timeout_s=decision_timeout_s,
                max_decisions=max_decisions,
                use_beam=use_beam,
                search_mode=search_mode,
                beam_max_depth=beam_max_depth,
            )

    workers = [asyncio.create_task(_worker()) for _ in range(min(concurrency, num_runs))]
    await asyncio.gather(*workers)
    return [result for result in results if result is not None]


def summarize_floor_reach(results: list[FloorReachResult]) -> dict[str, Any]:
    """Return aggregate floor and outcome statistics for one batch."""
    floors = [result.max_total_floor for result in results]
    errored = [result for result in results if result.error is not None]
    outcome_counts = Counter(result.outcome for result in results if result.error is None)
    floor_stats = None
    if floors:
        floor_stats = {
            "mean": statistics.fmean(floors),
            "median": statistics.median(floors),
            "min": min(floors),
            "max": max(floors),
            "stdev": statistics.pstdev(floors) if len(floors) > 1 else 0.0,
        }
    return {
        "runs_requested": len(results),
        "runs_errored": len(errored),
        "outcome_counts": dict(outcome_counts),
        "floor_stats": floor_stats,
        "floors": floors,
        "errors": [{"run_id": result.run_id, "error": result.error} for result in errored],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser)
    parser.add_argument("--character-id", required=True)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--num-runs", type=_positive_int, required=True)
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=1,
        help="parallel runs; use >1 only with a server that hosts independent instances",
    )
    parser.add_argument(
        "--no-beam",
        action="store_true",
        help="disable beam search and use HeuristicCombatSelector only",
    )
    parser.add_argument("--output", type=Path, default=None, help="write full JSON results here")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    results = await run_floor_reach_eval(
        character_id=args.character_id,
        ascension=args.ascension,
        num_runs=args.num_runs,
        concurrency=args.concurrency,
        use_beam=not args.no_beam,
        host=args.host,
        port=args.port,
        connect_timeout_s=args.connect_timeout,
        decision_timeout_s=args.decision_timeout,
        max_decisions=args.max_decisions,
        search_mode=args.search_mode,
        beam_max_depth=args.beam_depth,
    )
    summary = summarize_floor_reach(results)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "results": [vars(result) for result in results],
        }
        args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging(args.log_level)
    summary = asyncio.run(_run(args))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
