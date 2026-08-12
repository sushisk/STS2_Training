"""Fixed-seed A/B evaluation for stable-frontier pruners.

Each seed is run as two independent Combat episodes with identical scenario, policy/value
construction, BeamSearchConfig, and emulator seed. The only intended difference is the
``StableFrontierPruner`` implementation.

``action_id`` is decision-local and therefore is never used to compare separate A/B
instances. Common-prefix/divergence reporting uses a canonicalized legal-action payload
with ``action_id``/``is_available`` removed. After the first semantic divergence the states
may differ, so later behavior is compared only through arm-level outcomes and search-cost
metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any

from sts2_training.api import AsyncTrainingApiClient, TcpConnection
from sts2_training.api.contract import ROOT_BRANCH_ID
from sts2_training.api.transport import TransportError
from sts2_training.decision.beam_search import BeamSearchConfig, BeamSearchResult
from sts2_training.decision.combat_decision import COMBAT_BEAM_ACTION_TYPES
from sts2_training.decision.engine import CombatDecisionEngine, DecisionOutcome
from sts2_training.decision.learned_pruner import LinearStableFrontierPruner
from sts2_training.decision.search_modes import resolve_search_mode
from sts2_training.decision.stable_pruner import StableFrontierPruner, ValueTopKPruner
from sts2_training.runner._cli import add_common_arguments, configure_logging
from sts2_training.runner.scenario import CombatScenario, EnemyScenario
from sts2_training.selection.heuristic_selector import NoAvailableActionError

JsonObject = dict[str, Any]
EngineFactory = Callable[[Any, StableFrontierPruner, BeamSearchConfig], Any]

AB_REPORT_SCHEMA_VERSION = 1
_ARM_ORDERS = ("alternate", "baseline-first", "learned-first")
_VOLATILE_ACTION_FIELDS = frozenset({"action_id", "is_available"})


@dataclass(frozen=True)
class StablePrunerABDecision:
    index: int
    decision_point_id: str
    chosen_action_id: str
    chosen_action_type: str | None
    action_semantics: JsonObject
    action_signature: str
    source: str
    beam_reason: str | None
    beam_best_value: float | None
    nodes_expanded: int
    branches_created: int
    beam_total_ms: float


@dataclass(frozen=True)
class StablePrunerABArmResult:
    arm: str
    seed: int
    pruner_name: str
    pruner_version: str
    outcome: str | None
    decisions_made: int
    elapsed_s: float
    beam_decisions: int
    nodes_expanded: int
    branches_created: int
    beam_total_ms: float
    heuristic_fallbacks: int
    decisions: tuple[StablePrunerABDecision, ...]


@dataclass(frozen=True)
class StablePrunerABPairResult:
    seed: int
    baseline: StablePrunerABArmResult
    learned: StablePrunerABArmResult
    common_action_prefix: int
    first_divergence_index: int | None
    winner: str | None


@dataclass(frozen=True)
class StablePrunerABSummary:
    pairs: int
    resolved_outcome_pairs: int
    baseline_wins: int
    learned_wins: int
    outcome_ties: int
    unknown_outcome_pairs: int
    diverged_pairs: int
    mean_common_action_prefix: float
    baseline_mean_decisions: float
    learned_mean_decisions: float
    baseline_mean_elapsed_s: float
    learned_mean_elapsed_s: float
    baseline_mean_nodes_expanded: float
    learned_mean_nodes_expanded: float
    baseline_mean_beam_total_ms: float
    learned_mean_beam_total_ms: float
    learned_minus_baseline_mean_elapsed_s: float
    learned_minus_baseline_mean_nodes_expanded: float
    learned_minus_baseline_mean_beam_total_ms: float


@dataclass(frozen=True)
class StablePrunerABReport:
    schema_version: int
    scenario_template_sha256: str
    seeds: tuple[int, ...]
    search_config: dict[str, Any]
    arm_order: str
    learned_pruner_name: str
    learned_pruner_version: str
    learned_artifact_sha256: str | None
    pairs: tuple[StablePrunerABPairResult, ...]
    summary: StablePrunerABSummary


class StablePrunerABRunner:
    """Run paired Combat episodes while changing only the stable-frontier pruner."""

    def __init__(
        self,
        client: Any,
        *,
        scenario: CombatScenario,
        learned_pruner: StableFrontierPruner,
        beam_config: BeamSearchConfig,
        start_timeout_s: float = 30.0,
        decision_timeout_s: float = 30.0,
        max_decisions: int | None = None,
        arm_order: str = "alternate",
        engine_factory: EngineFactory | None = None,
    ) -> None:
        _validate_positive_timeout("start_timeout_s", start_timeout_s)
        _validate_positive_timeout("decision_timeout_s", decision_timeout_s)
        _validate_max_decisions(max_decisions)
        if arm_order not in _ARM_ORDERS:
            raise ValueError(f"arm_order must be one of {_ARM_ORDERS}")
        self._client = client
        self._scenario = scenario
        self._learned_pruner = learned_pruner
        self._beam_config = _clone_beam_config(beam_config)
        self._start_timeout_s = float(start_timeout_s)
        self._decision_timeout_s = float(decision_timeout_s)
        self._max_decisions = max_decisions
        self._arm_order = arm_order
        self._engine_factory = engine_factory or _default_engine_factory

    async def run(self, seeds: Sequence[int]) -> StablePrunerABReport:
        normalized_seeds = _validate_seeds(seeds)
        pairs: list[StablePrunerABPairResult] = []
        for pair_index, seed in enumerate(normalized_seeds):
            order = self._order_for_pair(pair_index)
            arm_results: dict[str, StablePrunerABArmResult] = {}
            for arm in order:
                pruner: StableFrontierPruner
                if arm == "baseline":
                    pruner = ValueTopKPruner()
                else:
                    pruner = self._learned_pruner
                arm_results[arm] = await self._run_arm(arm=arm, seed=seed, pruner=pruner)
            pairs.append(
                compare_arm_results(
                    seed=seed,
                    baseline=arm_results["baseline"],
                    learned=arm_results["learned"],
                )
            )
        pair_tuple = tuple(pairs)
        return StablePrunerABReport(
            schema_version=AB_REPORT_SCHEMA_VERSION,
            scenario_template_sha256=_scenario_template_sha256(self._scenario),
            seeds=normalized_seeds,
            search_config=_beam_config_record(self._beam_config),
            arm_order=self._arm_order,
            learned_pruner_name=str(
                getattr(self._learned_pruner, "name", type(self._learned_pruner).__name__)
            ),
            learned_pruner_version=str(getattr(self._learned_pruner, "version", "unknown")),
            learned_artifact_sha256=_artifact_sha256(self._learned_pruner),
            pairs=pair_tuple,
            summary=summarize_pairs(pair_tuple),
        )

    def _order_for_pair(self, pair_index: int) -> tuple[str, str]:
        if self._arm_order == "baseline-first":
            return ("baseline", "learned")
        if self._arm_order == "learned-first":
            return ("learned", "baseline")
        if pair_index % 2 == 0:
            return ("baseline", "learned")
        return ("learned", "baseline")

    async def _run_arm(
        self,
        *,
        arm: str,
        seed: int,
        pruner: StableFrontierPruner,
    ) -> StablePrunerABArmResult:
        config = _clone_beam_config(self._beam_config)
        engine = self._engine_factory(self._client, pruner, config)
        scenario = replace(self._scenario, seed=seed)
        instance_id = await self._client.start_instance(
            scenario.to_instance_config(), timeout_s=self._start_timeout_s
        )
        t0 = time.monotonic()
        decisions: list[StablePrunerABDecision] = []
        next_decision: JsonObject | None = None
        final_dto: JsonObject = {}
        try:
            while True:
                if next_decision is None:
                    raw_decision = await self._client.get_decision(
                        instance_id,
                        ROOT_BRANCH_ID,
                        timeout_s=self._decision_timeout_s,
                    )
                else:
                    raw_decision = next_decision
                decision, dto = _validated_decision(raw_decision)
                final_dto = dict(dto)
                if _is_terminal(dto):
                    break

                legal_actions = _available_actions_by_id(dto)
                if not legal_actions:
                    raise NoAvailableActionError(
                        "non-terminal A/B decision has no available legal action",
                        decision=decision,
                    )

                deadline = time.monotonic() + self._decision_timeout_s
                remaining = deadline - time.monotonic()
                outcome: DecisionOutcome = await engine.decide(
                    instance_id,
                    timeout_s=remaining,
                    decision=decision,
                )
                chosen = outcome.chosen_action_id
                if chosen is None or chosen not in legal_actions:
                    raise NoAvailableActionError(
                        "A/B engine did not return an available action",
                        decision=decision,
                    )
                decisions.append(
                    _decision_metric(
                        index=len(decisions),
                        decision_point_id=decision["decision_point_id"],
                        chosen_action_id=chosen,
                        chosen_action=legal_actions[chosen],
                        outcome=outcome,
                    )
                )

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TransportError("A/B decision timeout elapsed before commit_action")
                response = await self._client.commit_action(
                    instance_id,
                    decision["decision_point_id"],
                    chosen,
                    timeout_s=remaining,
                )
                next_decision, next_dto = _validated_decision(response)
                final_dto = dict(next_dto)
                if _is_terminal(next_dto):
                    break
                if self._max_decisions is not None and len(decisions) >= self._max_decisions:
                    raise RuntimeError(
                        f"A/B arm={arm} seed={seed} reached max_decisions="
                        f"{self._max_decisions} without terminal state"
                    )
        finally:
            await _close_best_effort(
                self._client,
                instance_id,
                timeout_s=min(10.0, self._decision_timeout_s),
            )

        beam_decisions = sum(1 for item in decisions if item.beam_reason is not None)
        return StablePrunerABArmResult(
            arm=arm,
            seed=seed,
            pruner_name=str(getattr(pruner, "name", type(pruner).__name__)),
            pruner_version=str(getattr(pruner, "version", "unknown")),
            outcome=_outcome_name(final_dto),
            decisions_made=len(decisions),
            elapsed_s=time.monotonic() - t0,
            beam_decisions=beam_decisions,
            nodes_expanded=sum(item.nodes_expanded for item in decisions),
            branches_created=sum(item.branches_created for item in decisions),
            beam_total_ms=sum(item.beam_total_ms for item in decisions),
            heuristic_fallbacks=sum(
                1 for item in decisions if item.source == "heuristic_fallback"
            ),
            decisions=tuple(decisions),
        )


def compare_arm_results(
    *,
    seed: int,
    baseline: StablePrunerABArmResult,
    learned: StablePrunerABArmResult,
) -> StablePrunerABPairResult:
    baseline_actions = tuple(item.action_signature for item in baseline.decisions)
    learned_actions = tuple(item.action_signature for item in learned.decisions)
    common_prefix = 0
    for baseline_action, learned_action in zip(baseline_actions, learned_actions, strict=False):
        if baseline_action != learned_action:
            break
        common_prefix += 1
    if common_prefix < min(len(baseline_actions), len(learned_actions)):
        divergence: int | None = common_prefix
    elif len(baseline_actions) != len(learned_actions):
        divergence = common_prefix
    else:
        divergence = None

    baseline_score = _outcome_score(baseline.outcome)
    learned_score = _outcome_score(learned.outcome)
    winner: str | None
    if baseline_score is None or learned_score is None:
        winner = None
    elif learned_score > baseline_score:
        winner = "learned"
    elif baseline_score > learned_score:
        winner = "baseline"
    else:
        winner = "tie"

    return StablePrunerABPairResult(
        seed=seed,
        baseline=baseline,
        learned=learned,
        common_action_prefix=common_prefix,
        first_divergence_index=divergence,
        winner=winner,
    )


def summarize_pairs(pairs: Sequence[StablePrunerABPairResult]) -> StablePrunerABSummary:
    if not pairs:
        raise ValueError("pairs must not be empty")
    resolved = [pair for pair in pairs if pair.winner is not None]
    return StablePrunerABSummary(
        pairs=len(pairs),
        resolved_outcome_pairs=len(resolved),
        baseline_wins=sum(pair.winner == "baseline" for pair in pairs),
        learned_wins=sum(pair.winner == "learned" for pair in pairs),
        outcome_ties=sum(pair.winner == "tie" for pair in pairs),
        unknown_outcome_pairs=sum(pair.winner is None for pair in pairs),
        diverged_pairs=sum(pair.first_divergence_index is not None for pair in pairs),
        mean_common_action_prefix=mean(pair.common_action_prefix for pair in pairs),
        baseline_mean_decisions=mean(pair.baseline.decisions_made for pair in pairs),
        learned_mean_decisions=mean(pair.learned.decisions_made for pair in pairs),
        baseline_mean_elapsed_s=mean(pair.baseline.elapsed_s for pair in pairs),
        learned_mean_elapsed_s=mean(pair.learned.elapsed_s for pair in pairs),
        baseline_mean_nodes_expanded=mean(pair.baseline.nodes_expanded for pair in pairs),
        learned_mean_nodes_expanded=mean(pair.learned.nodes_expanded for pair in pairs),
        baseline_mean_beam_total_ms=mean(pair.baseline.beam_total_ms for pair in pairs),
        learned_mean_beam_total_ms=mean(pair.learned.beam_total_ms for pair in pairs),
        learned_minus_baseline_mean_elapsed_s=mean(
            pair.learned.elapsed_s - pair.baseline.elapsed_s for pair in pairs
        ),
        learned_minus_baseline_mean_nodes_expanded=mean(
            pair.learned.nodes_expanded - pair.baseline.nodes_expanded for pair in pairs
        ),
        learned_minus_baseline_mean_beam_total_ms=mean(
            pair.learned.beam_total_ms - pair.baseline.beam_total_ms for pair in pairs
        ),
    )


def _decision_metric(
    *,
    index: int,
    decision_point_id: str,
    chosen_action_id: str,
    chosen_action: Mapping[str, Any],
    outcome: DecisionOutcome,
) -> StablePrunerABDecision:
    semantics = _action_semantics(chosen_action)
    action_type = chosen_action.get("action_type")
    chosen_action_type = action_type if isinstance(action_type, str) and action_type else None
    result: BeamSearchResult | None = outcome.beam_result
    common = {
        "index": index,
        "decision_point_id": decision_point_id,
        "chosen_action_id": chosen_action_id,
        "chosen_action_type": chosen_action_type,
        "action_semantics": semantics,
        "action_signature": _action_signature(semantics),
        "source": outcome.source,
    }
    if result is None:
        return StablePrunerABDecision(
            **common,
            beam_reason=None,
            beam_best_value=None,
            nodes_expanded=0,
            branches_created=0,
            beam_total_ms=0.0,
        )
    return StablePrunerABDecision(
        **common,
        beam_reason=result.reason,
        beam_best_value=result.best_value,
        nodes_expanded=result.stats.nodes_expanded,
        branches_created=result.stats.branches_created,
        beam_total_ms=result.stats.total_ms,
    )


def _action_semantics(action: Mapping[str, Any]) -> JsonObject:
    return {
        str(key): _jsonable(value)
        for key, value in action.items()
        if key not in _VOLATILE_ACTION_FIELDS
    }


def _action_signature(semantics: Mapping[str, Any]) -> str:
    return json.dumps(
        semantics,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _scenario_template_sha256(scenario: CombatScenario) -> str:
    payload = scenario.to_instance_config()
    payload.pop("seed", None)
    encoded = json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_sha256(pruner: StableFrontierPruner) -> str | None:
    metadata = getattr(pruner, "artifact_metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("artifact_sha256")
    return value if isinstance(value, str) and value else None


def _default_engine_factory(
    client: Any,
    pruner: StableFrontierPruner,
    config: BeamSearchConfig,
) -> CombatDecisionEngine:
    return CombatDecisionEngine(
        client,
        stable_pruner=pruner,
        beam_config=_clone_beam_config(config),
    )


def _clone_beam_config(config: BeamSearchConfig) -> BeamSearchConfig:
    return replace(
        config,
        beam_searchable_action_types=frozenset(config.beam_searchable_action_types),
        simulation_options=(
            None if config.simulation_options is None else dict(config.simulation_options)
        ),
    )


def _beam_config_record(config: BeamSearchConfig) -> dict[str, Any]:
    return {
        "beam_width": config.beam_width,
        "top_k_actions": config.top_k_actions,
        "max_depth": config.max_depth,
        "time_budget_ms": config.time_budget_ms,
        "max_batch_size": config.max_batch_size,
        "expand_partial": config.expand_partial,
        "release_branches_on_finish": config.release_branches_on_finish,
        "beam_searchable_action_types": sorted(config.beam_searchable_action_types),
        "max_continuation_steps": config.max_continuation_steps,
        "simulation_options": (
            None if config.simulation_options is None else dict(config.simulation_options)
        ),
    }


def _validated_decision(value: Any) -> tuple[JsonObject, Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise RuntimeError("A/B decision response must be a mapping")
    decision_point_id = value.get("decision_point_id")
    dto = value.get("masked_emulator_dto")
    if not isinstance(decision_point_id, str) or not decision_point_id:
        raise RuntimeError("A/B decision response has invalid decision_point_id")
    if not isinstance(dto, Mapping):
        raise RuntimeError("A/B decision response has invalid masked_emulator_dto")
    return dict(value), dto


def _available_actions_by_id(dto: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = dto.get("legal_actions")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for action in raw:
        if not isinstance(action, Mapping) or action.get("is_available") is False:
            continue
        action_id = action.get("action_id")
        if isinstance(action_id, str) and action_id:
            result[action_id] = action
    return result


def _is_terminal(dto: Mapping[str, Any]) -> bool:
    return dto.get("terminal") is True or dto.get("run_terminal") is True


def _outcome_name(dto: Mapping[str, Any]) -> str | None:
    value = dto.get("outcome")
    return value if isinstance(value, str) and value else None


def _outcome_score(outcome: str | None) -> int | None:
    if outcome is None:
        return None
    normalized = outcome.strip().lower()
    if normalized in {"victory", "win"}:
        return 1
    if normalized in {"defeat", "loss"}:
        return 0
    return None


async def _close_best_effort(client: Any, instance_id: str, *, timeout_s: float) -> None:
    if getattr(client, "pending_retry", None) is not None or getattr(
        client, "session_invalid", False
    ):
        return
    try:
        await client.close_instance(instance_id, timeout_s=timeout_s)
    except Exception:  # noqa: BLE001 - A/B result/error must not be masked by cleanup
        return


def _validate_positive_timeout(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")


def _validate_max_decisions(value: int | None) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
    ):
        raise ValueError("max_decisions must be a positive integer or None")


def _validate_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    if not seeds:
        raise ValueError("at least one seed is required")
    normalized: list[int] = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seeds must contain only integers")
        normalized.append(seed)
    return tuple(normalized)


def _parse_seed_list(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("must contain at least one integer seed")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a comma-separated list of integers") from exc


def _scenario_from_json(data: Mapping[str, Any]) -> CombatScenario:
    fields = dict(data)
    raw_enemies = fields.get("enemies")
    if not isinstance(raw_enemies, Sequence) or isinstance(
        raw_enemies, (str, bytes, bytearray)
    ):
        raise ValueError("scenario JSON must contain enemies as a sequence")
    fields["enemies"] = [EnemyScenario(**dict(enemy)) for enemy in raw_enemies]
    return CombatScenario(**fields)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=_parse_seed_list,
        default=None,
        help="comma-separated emulator seeds; default: use the seed in --scenario",
    )
    parser.add_argument(
        "--arm-order",
        choices=_ARM_ORDERS,
        default="alternate",
        help="alternate reduces systematic first/second-run ordering bias",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON report path; the same report is always printed to stdout",
    )
    return parser.parse_args(argv)


def _cli_beam_config(args: argparse.Namespace) -> BeamSearchConfig:
    config = resolve_search_mode(args.search_mode, max_depth=args.beam_depth)
    return replace(
        config,
        beam_searchable_action_types=COMBAT_BEAM_ACTION_TYPES,
        simulation_options=(
            None if config.simulation_options is None else dict(config.simulation_options)
        ),
    )


async def _run(args: argparse.Namespace) -> StablePrunerABReport:
    scenario_payload = json.loads(args.scenario.read_text(encoding="utf-8"))
    if not isinstance(scenario_payload, Mapping):
        raise ValueError("scenario JSON root must be an object")
    scenario = _scenario_from_json(scenario_payload)
    seeds = args.seeds if args.seeds is not None else (scenario.seed,)
    learned = LinearStableFrontierPruner.from_weights_file(args.weights)
    connection = TcpConnection(
        host=args.host,
        port=args.port,
        connect_timeout_s=args.connect_timeout,
    )
    async with AsyncTrainingApiClient(connection) as client:
        report = await StablePrunerABRunner(
            client,
            scenario=scenario,
            learned_pruner=learned,
            beam_config=_cli_beam_config(args),
            decision_timeout_s=args.decision_timeout,
            max_decisions=args.max_decisions,
            arm_order=args.arm_order,
        ).run(seeds)
    return report


def report_to_json(report: StablePrunerABReport) -> str:
    return json.dumps(asdict(report), ensure_ascii=False, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging(args.log_level)
    report = asyncio.run(_run(args))
    encoded = report_to_json(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
