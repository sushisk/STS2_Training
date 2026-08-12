"""Aggregate floor-reach evaluation harness for `HeuristicCombatSelector`/
`CombatDecisionEngine` policy changes, across many independently-random-seed Whole Runs.

Why this exists (see the investigation this replaced): comparing the SAME seed's floor
reach before/after a decision-policy change is not valid. A differently-scored decision
diverges the whole playthrough - different rooms visited, different fights, different RNG
draws consumed in a different order - so "seed X used to reach floor 22, now reaches
floor 8" is not "this seed got worse", it is a different game. Only aggregate statistics
(mean/median/stdev) across many independently random seeds are informative for evaluating
a policy change; single-seed or matched-seed comparisons should not be used.

CLI use::

    # Default policy (beam search, "standard" preset)
    python -m sts2_training.runner.floor_reach_eval --character-id IRONCLAD --num-runs 30

    # Heuristic-only (no beam search) - what data collection currently runs
    python -m sts2_training.runner.floor_reach_eval --character-id IRONCLAD --num-runs 30 --no-beam

Programmatic use::

    results = await run_floor_reach_eval(character_id="IRONCLAD", num_runs=30, use_beam=False)
    summary = summarize_floor_reach(results)
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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
    """One `start_new_run` attempt, tracked for the deepest `totalFloor` it reached."""

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
    act_index_at_max: "int | None" = None
    decisions_made: int = 0
    source_counts: dict[str, int] = field(default_factory=dict)


_current_state: "contextvars.ContextVar[_RunState | None]" = contextvars.ContextVar(
    "_floor_reach_eval_current_state", default=None
)
_orig_decide = engine_module.CombatDecisionEngine.decide
_patched = False


async def _tracking_decide(self: CombatDecisionEngine, *args: Any, **kwargs: Any) -> Any:
    outcome = await _orig_decide(self, *args, **kwargs)
    state = _current_state.get()
    if state is not None:
        state.decisions_made += 1
        state.source_counts[outcome.source] = state.source_counts.get(outcome.source, 0) + 1
        decision = outcome.decision
        if isinstance(decision, dict):
            dto = decision.get("masked_emulator_dto")
            if isinstance(dto, dict):
                total_floor = dto.get("totalFloor")
                if isinstance(total_floor, int) and total_floor > state.max_total_floor:
                    state.max_total_floor = total_floor
                    act_index = dto.get("currentActIndex")
                    state.act_index_at_max = act_index if isinstance(act_index, int) else None
    return outcome


def _ensure_patched() -> None:
    """Installs the tracking wrapper once, process-wide. Safe under `asyncio` concurrency:
    each `run_one` Task sets its own `_current_state` before awaiting `start_new_run`, and
    `ContextVar` writes are Task-local (a Task's context is copied at creation and never
    shared with siblings), so concurrent runs never see each other's counters."""
    global _patched
    if not _patched:
        engine_module.CombatDecisionEngine.decide = _tracking_decide
        _patched = True


def _build_engine(
    client: Any,
    *,
    seed: int,
    use_beam: bool,
    search_mode: "str | BeamSearchConfig | None",
    beam_max_depth: "int | None",
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
    max_decisions: "int | None",
    use_beam: bool,
    search_mode: "str | BeamSearchConfig | None",
    beam_max_depth: "int | None",
) -> FloorReachResult:
    _ensure_patched()
    state = _RunState()
    token = _current_state.set(state)
    t0 = time.monotonic()
    outcome_label: "str | None" = None
    error: "str | None" = None
    client: "AsyncTrainingApiClient | None" = None
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
        final_total_floor = result.final_dto.get("totalFloor")
        if isinstance(final_total_floor, int) and final_total_floor > state.max_total_floor:
            state.max_total_floor = final_total_floor
    except Exception as exc:  # noqa: BLE001 - one run's failure must not sink the batch
        _LOG.exception("floor-reach-eval run %s failed", run_id)
        error = f"{type(exc).__name__}: {exc}"
    finally:
        _current_state.reset(token)
        if client is not None:
            try:
                await client.close()
            except Exception:  # noqa: BLE001 - transport cleanup must not overwrite the result
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
    connection_factory: "Callable[[], Any] | None" = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    connect_timeout_s: float = 5.0,
    decision_timeout_s: float = 90.0,
    max_decisions: "int | None" = 600,
    search_mode: "str | BeamSearchConfig | None" = None,
    beam_max_depth: "int | None" = None,
) -> list[FloorReachResult]:
    """Run `num_runs` independent, independently-random-seed `start_new_run` episodes
    with bounded concurrency, tracking the deepest `totalFloor` each one reaches (even
    if it never terminates within `max_decisions` - the point of this harness is
    depth reached, not completion). See the module docstring for why matched/fixed
    seeds must not be used to compare two policies here.

    `concurrency` defaults to 1: confirmed live that `API.tcp_server`'s plain TCP server
    backs every `instance_id` with the same one-`GameInstance`-per-process Emulator, so
    concurrent runs against it supersede each other mid-flight (Emulator raises
    `InvalidOperationException: ...has been superseded by a later GameInstance...`) -
    only raise `concurrency` above 1 against a server that actually hosts multiple
    simultaneous instances (e.g. a `WholeRunWorkerPool`-backed one), never the plain one.
    """
    if not isinstance(num_runs, int) or isinstance(num_runs, bool) or num_runs <= 0:
        raise ValueError("num_runs must be a positive integer")
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency <= 0:
        raise ValueError("concurrency must be a positive integer")
    if not isinstance(character_id, str) or not character_id.strip():
        raise ValueError("character_id must be a non-empty string")
    resolve_search_mode(search_mode, max_depth=beam_max_depth)  # validate early

    if connection_factory is None:
        factory: Callable[[], Any] = lambda: TcpConnection(
            host=host, port=port, connect_timeout_s=connect_timeout_s
        )
    else:
        factory = connection_factory

    batch_tag = f"{character_id.lower()}-{int(time.time())}"
    results: "list[FloorReachResult | None]" = [None] * num_runs
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
    return [r for r in results if r is not None]


def summarize_floor_reach(results: list[FloorReachResult]) -> dict[str, Any]:
    """Aggregate stats over one batch. `floor_stats` is `None` when every run errored
    before reaching any tracked decision (nothing to summarize)."""
    floors = [r.max_total_floor for r in results]
    errored = [r for r in results if r.error is not None]
    outcome_counts = Counter(r.outcome for r in results if r.error is None)
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
        "errors": [{"run_id": r.run_id, "error": r.error} for r in errored],
    }


def _parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
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
        help="only raise above 1 against a server that hosts multiple simultaneous "
        "instances - the plain API.tcp_server backs every instance_id with the same "
        "single Emulator GameInstance, so concurrent runs supersede each other",
    )
    parser.add_argument(
        "--no-beam",
        action="store_true",
        help="disable beam search entirely (HeuristicCombatSelector only) - "
        "what data collection currently runs; --search-mode/--beam-depth are ignored",
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
            "results": [vars(r) for r in results],
        }
        args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return summary


def main(argv: "list[str] | None" = None) -> int:
    args = _parse_args(argv)
    configure_logging(args.log_level)
    summary = asyncio.run(_run(args))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
