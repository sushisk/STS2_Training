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
import json
import logging
import random
import statistics
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sts2_training.api import AsyncTrainingApiClient, TcpConnection
from sts2_training.decision import CombatDecisionEngine, DecisionOutcome
from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.decision.policy import PolicyModel
from sts2_training.decision.search_modes import resolve_search_mode
from sts2_training.decision.search_trace import InMemorySearchTraceCollector
from sts2_training.decision.stable_pruner import StableFrontierPruner
from sts2_training.decision.value import ValueModel
from sts2_training.runner._cli import _positive_int, add_common_arguments, configure_logging
from sts2_training.runner.beam_scope import runner_combat_beam_config
from sts2_training.runner.start_new_run import start_new_run
from sts2_training.selection.heuristic_selector import HeuristicCombatSelector
from sts2_training.run_log import JsonlRunEventLogger

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


def _record_floor(state: _RunState, dto: Mapping[str, Any]) -> None:
    total_floor = dto.get("totalFloor")
    if not isinstance(total_floor, int) or total_floor <= state.max_total_floor:
        return

    state.max_total_floor = total_floor
    act_index = dto.get("currentActIndex")
    state.act_index_at_max = act_index if isinstance(act_index, int) else None


class _TrackingCombatDecisionEngine(CombatDecisionEngine):
    """Combat engine that records decision metadata for one evaluation run."""

    def __init__(
        self,
        client: Any,
        *,
        state: _RunState,
        action_score_policy: Any | None = None,
        action_score_logger: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._score_trace_collector = (
            InMemorySearchTraceCollector() if action_score_logger is not None else None
        )
        if self._score_trace_collector is not None:
            kwargs["trace_collector"] = self._score_trace_collector
        super().__init__(client, **kwargs)
        self._state = state
        self._action_score_policy = action_score_policy
        self._action_score_logger = action_score_logger

    async def decide(
        self,
        instance_id: str,
        *,
        timeout_s: float,
        decision: Mapping[str, Any] | None = None,
    ) -> DecisionOutcome:
        if self._score_trace_collector is not None:
            self._score_trace_collector.events.clear()
        outcome = await super().decide(instance_id, timeout_s=timeout_s, decision=decision)
        self._log_action_scores(instance_id, outcome)
        self._log_score_trace(instance_id, outcome)
        self._state.decisions_made += 1
        self._state.source_counts[outcome.source] += 1

        dto = outcome.decision.get("masked_emulator_dto")
        if isinstance(dto, Mapping):
            _record_floor(self._state, dto)
        return outcome

    def _log_score_trace(self, instance_id: str, outcome: DecisionOutcome) -> None:
        collector = self._score_trace_collector
        logger = self._action_score_logger
        if collector is None or logger is None:
            return
        for event in collector.events:
            try:
                trace_event = asdict(event)
                if hasattr(event, "nodes") and hasattr(event, "to_prune_context"):
                    views = event.node_views()
                    pruner = self.beam_search.stable_pruner
                    score_fn = getattr(pruner, "frontier_scores", None)
                    if score_fn is None:
                        score_fn = getattr(pruner, "score_batch", None)
                    if score_fn is not None:
                        frontier_scores = score_fn(
                            views,
                            context=event.to_prune_context(),
                        )
                        for node, frontier_score in zip(
                            trace_event.get("nodes", []), frontier_scores, strict=True
                        ):
                            node["frontier_score"] = frontier_score
                logger(
                    {
                        "event": "score_trace",
                        "instance_id": instance_id,
                        "decision_point_id": outcome.decision.get("decision_point_id"),
                        "decision_source": outcome.source,
                        "selected_action_id": outcome.chosen_action_id,
                        "trace_event": trace_event,
                    }
                )
            except Exception:
                _LOG.exception("score-trace logging failed")

    def _log_action_scores(self, instance_id: str, outcome: DecisionOutcome) -> None:
        policy = self._action_score_policy
        logger = self._action_score_logger
        if policy is None or logger is None or not hasattr(policy, "score_action"):
            return
        dto = outcome.decision.get("masked_emulator_dto")
        if not isinstance(dto, Mapping) or dto.get("boundary") != "stable":
            return
        actions = dto.get("legal_actions")
        if not isinstance(actions, list):
            return
        try:
            scored = []
            for action in actions:
                if not isinstance(action, Mapping) or action.get("is_available") is False:
                    continue
                score = policy.score_action(action, dto)
                scored.append(
                    {
                        "action_id": action.get("action_id"),
                        "action_type": action.get("action_type"),
                        "label": action.get("label"),
                        "score": score,
                    }
                )
            scored.sort(key=lambda item: (-item["score"], str(item["action_id"])))
            logger(
                {
                    "event": "action_score",
                    "instance_id": instance_id,
                    "decision_point_id": outcome.decision.get("decision_point_id"),
                    "boundary": dto.get("boundary"),
                    "total_floor": dto.get("totalFloor"),
                    "selected_action_id": outcome.chosen_action_id,
                    "decision_source": outcome.source,
                    "candidates": scored,
                }
            )
        except Exception:
            _LOG.exception("action-score logging failed")


def _build_engine(
    client: Any,
    *,
    state: _RunState,
    seed: int,
    use_beam: bool,
    search_mode: str | BeamSearchConfig | None,
    beam_max_depth: int | None,
    policy: PolicyModel | None = None,
    value_fn: ValueModel | None = None,
    stable_pruner: StableFrontierPruner | None = None,
    action_score_policy: Any | None = None,
    action_score_logger: Any | None = None,
) -> CombatDecisionEngine:
    fallback_selector = HeuristicCombatSelector(random.Random(seed))
    if use_beam:
        # Named/default presets carry the conservative dataclass scope; without widening,
        # the engine drops continuation branches (e.g. targeted cards in multi-enemy
        # fights). An explicit BeamSearchConfig remains caller-authoritative.
        beam_config = resolve_search_mode(search_mode, max_depth=beam_max_depth)
        if not isinstance(search_mode, BeamSearchConfig):
            beam_config = runner_combat_beam_config(beam_config)
        return _TrackingCombatDecisionEngine(
            client,
            state=state,
            policy=policy,
            value_fn=value_fn,
            stable_pruner=stable_pruner,
            action_score_policy=action_score_policy,
            action_score_logger=action_score_logger,
            beam_config=beam_config,
            fallback_selector=fallback_selector,
        )
    return _TrackingCombatDecisionEngine(
        client,
        state=state,
        policy=policy,
        value_fn=value_fn,
        stable_pruner=stable_pruner,
        action_score_policy=action_score_policy,
        action_score_logger=action_score_logger,
        beam_action_types=frozenset(),
        fallback_selector=fallback_selector,
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
    policy: PolicyModel | None,
    value_fn: ValueModel | None,
    stable_pruner: StableFrontierPruner | None,
    selection_log_dir: Path | None,
    action_score_log_dir: Path | None,
) -> FloorReachResult:
    state = _RunState()
    t0 = time.monotonic()
    outcome_label: str | None = None
    error: str | None = None
    client: AsyncTrainingApiClient | None = None
    selection_logger: JsonlRunEventLogger | None = None
    action_score_logger: JsonlRunEventLogger | None = None
    try:
        connection = connection_factory()
        if selection_log_dir is not None:
            selection_log_dir.mkdir(parents=True, exist_ok=True)
            selection_logger = JsonlRunEventLogger(
                selection_log_dir / f"{run_id}.jsonl", append=False
            )
        if action_score_log_dir is not None:
            action_score_log_dir.mkdir(parents=True, exist_ok=True)
            action_score_logger = JsonlRunEventLogger(
                action_score_log_dir / f"{run_id}.jsonl", append=False
            )
        client = AsyncTrainingApiClient(connection, selection_logger=selection_logger)
        await connection.connect()
        engine = _build_engine(
            client,
            state=state,
            seed=seed,
            use_beam=use_beam,
            search_mode=search_mode,
            beam_max_depth=beam_max_depth,
            policy=policy,
            value_fn=value_fn,
            stable_pruner=stable_pruner,
            action_score_policy=policy,
            action_score_logger=action_score_logger,
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
        if client is not None:
            try:
                await client.close()
            except Exception:  # noqa: BLE001 - cleanup must not replace the run result
                _LOG.exception("floor-reach-eval run %s: client close failed", run_id)
        if selection_logger is not None:
            try:
                selection_logger.close()
            except Exception:  # noqa: BLE001 - cleanup must not replace the run result
                _LOG.exception("floor-reach-eval run %s: selection log close failed", run_id)
        if action_score_logger is not None:
            try:
                action_score_logger.close()
            except Exception:  # noqa: BLE001 - cleanup must not replace the run result
                _LOG.exception("floor-reach-eval run %s: action-score log close failed", run_id)

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
    policy: PolicyModel | None = None,
    value_fn: ValueModel | None = None,
    stable_pruner: StableFrontierPruner | None = None,
    selection_log_dir: Path | None = None,
    action_score_log_dir: Path | None = None,
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
                policy=policy,
                value_fn=value_fn,
                stable_pruner=stable_pruner,
                selection_log_dir=selection_log_dir,
                action_score_log_dir=action_score_log_dir,
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
