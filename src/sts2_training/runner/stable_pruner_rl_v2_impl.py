"""Collect paired on-policy trajectories for stable-pruner RL fine-tuning.

Each seed is evaluated with the existing fixed-seed A/B harness. The baseline arm uses
``ValueTopKPruner`` while the learned arm uses a stochastic Plackett-Luce wrapper around
the current linear artifact. The paired terminal-outcome/search-cost delta becomes the
episode reward for the learned arm's pruning decisions.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any

from sts2_training.api import AsyncTrainingApiClient, TcpConnection
from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.decision.combat_decision import COMBAT_BEAM_ACTION_TYPES
from sts2_training.decision.learned_pruner import LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION
from sts2_training.decision.pruner_features import PRUNER_FEATURE_SCHEMA_VERSION
from sts2_training.decision.pruner_rl import (
    InMemoryPrunerRLStepCollector,
    PlackettLuceLinearStableFrontierPruner,
    PrunerRLStep,
)
from sts2_training.decision.search_modes import resolve_search_mode
from sts2_training.decision.stable_pruner import STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION
from sts2_training.runner._cli import add_common_arguments, configure_logging
from sts2_training.runner.scenario import CombatScenario, EnemyScenario
from sts2_training.runner.stable_pruner_ab import (
    StablePrunerABPairResult,
    StablePrunerABRunner,
)

RL_TRAJECTORY_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class PairedPrunerReward:
    outcome_delta: float
    nodes_expanded_delta: int
    beam_ms_delta: float
    node_cost_weight: float
    beam_ms_cost_weight: float
    total: float


@dataclass(frozen=True)
class StablePrunerRLCollectionSummary:
    seeds_requested: int
    trajectories_written: int
    skipped_unknown_outcome: int
    skipped_no_pruning_choice: int
    mean_reward: float | None


def paired_pruner_reward(
    pair: StablePrunerABPairResult,
    *,
    node_cost_weight: float = 0.0,
    beam_ms_cost_weight: float = 0.0,
) -> PairedPrunerReward | None:
    """Return paired learned-minus-baseline reward, or None for unknown outcomes."""

    _validate_non_negative_finite("node_cost_weight", node_cost_weight)
    _validate_non_negative_finite("beam_ms_cost_weight", beam_ms_cost_weight)
    baseline_score = _outcome_score(pair.baseline.outcome)
    learned_score = _outcome_score(pair.learned.outcome)
    if baseline_score is None or learned_score is None:
        return None
    nodes_delta = pair.learned.nodes_expanded - pair.baseline.nodes_expanded
    beam_ms_delta = pair.learned.beam_total_ms - pair.baseline.beam_total_ms
    outcome_delta = learned_score - baseline_score
    total = (
        outcome_delta
        - float(node_cost_weight) * nodes_delta
        - float(beam_ms_cost_weight) * beam_ms_delta
    )
    return PairedPrunerReward(
        outcome_delta=outcome_delta,
        nodes_expanded_delta=nodes_delta,
        beam_ms_delta=beam_ms_delta,
        node_cost_weight=float(node_cost_weight),
        beam_ms_cost_weight=float(beam_ms_cost_weight),
        total=total,
    )


def rl_episode_record(
    *,
    seed: int,
    sampler_seed: int,
    report: Any,
    reward: PairedPrunerReward,
    pruner: PlackettLuceLinearStableFrontierPruner,
    steps: Sequence[PrunerRLStep],
) -> dict[str, Any]:
    """Build one auditable on-policy trajectory record."""

    if not steps:
        raise ValueError("RL trajectory requires at least one stochastic pruning choice")
    metadata = pruner.artifact_metadata
    artifact_sha = metadata.get("artifact_sha256")
    if not isinstance(artifact_sha, str) or not artifact_sha:
        raise ValueError("behavior pruner must come from a hashed learned artifact")
    _require_schema(
        metadata,
        "artifact_schema_version",
        LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION,
        "behavior artifact",
    )
    _require_schema(
        metadata,
        "stable_prune_node_view_schema_version",
        STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
        "behavior artifact",
    )
    _require_schema(
        metadata,
        "feature_schema_version",
        PRUNER_FEATURE_SCHEMA_VERSION,
        "behavior artifact",
    )
    for step in steps:
        if step.stable_prune_node_view_schema_version != STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION:
            raise ValueError("RL step stable-prune node-view schema mismatch")
        if step.feature_schema_version != PRUNER_FEATURE_SCHEMA_VERSION:
            raise ValueError("RL step feature schema mismatch")

    pair = report.pairs[0]
    return {
        "record_type": "stable_pruner_rl_episode",
        "record_schema_version": RL_TRAJECTORY_SCHEMA_VERSION,
        "episode_id": f"{report.scenario_template_sha256[:16]}:{seed}:{sampler_seed}",
        "seed": seed,
        "scenario_template_sha256": report.scenario_template_sha256,
        "search_config": report.search_config,
        "behavior": {
            "pruner_name": pruner.name,
            "pruner_version": pruner.version,
            "artifact_sha256": artifact_sha,
            "artifact_schema_version": LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION,
            "stable_prune_node_view_schema_version": STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
            "feature_schema_version": PRUNER_FEATURE_SCHEMA_VERSION,
            "temperature": pruner.temperature,
            "sampler_seed": sampler_seed,
        },
        "reward": asdict(reward),
        "paired_result": {
            "baseline_outcome": pair.baseline.outcome,
            "learned_outcome": pair.learned.outcome,
            "winner": pair.winner,
            "baseline_nodes_expanded": pair.baseline.nodes_expanded,
            "learned_nodes_expanded": pair.learned.nodes_expanded,
            "baseline_beam_total_ms": pair.baseline.beam_total_ms,
            "learned_beam_total_ms": pair.learned.beam_total_ms,
            "common_action_prefix": pair.common_action_prefix,
            "first_divergence_index": pair.first_divergence_index,
        },
        "steps": [asdict(step) for step in steps],
    }


async def collect_rl_trajectories(
    client: Any,
    *,
    scenario: CombatScenario,
    weights: Path,
    seeds: Sequence[int],
    beam_config: BeamSearchConfig,
    temperature: float,
    sampler_seed: int,
    node_cost_weight: float,
    beam_ms_cost_weight: float,
    decision_timeout_s: float,
    max_decisions: int | None,
) -> tuple[list[dict[str, Any]], StablePrunerRLCollectionSummary]:
    normalized_seeds = _validate_seeds(seeds)
    records: list[dict[str, Any]] = []
    rewards: list[float] = []
    skipped_unknown = 0
    skipped_no_choice = 0

    for index, seed in enumerate(normalized_seeds):
        behavior_seed = _derived_sampler_seed(sampler_seed, seed=seed, index=index)
        collector = InMemoryPrunerRLStepCollector()
        pruner = PlackettLuceLinearStableFrontierPruner.from_weights_file(
            weights,
            temperature=temperature,
            seed=behavior_seed,
            collector=collector,
        )
        arm_order = "baseline-first" if index % 2 == 0 else "learned-first"
        report = await StablePrunerABRunner(
            client,
            scenario=scenario,
            learned_pruner=pruner,
            beam_config=_clone_beam_config(beam_config),
            decision_timeout_s=decision_timeout_s,
            max_decisions=max_decisions,
            arm_order=arm_order,
        ).run([seed])
        pair = report.pairs[0]
        reward = paired_pruner_reward(
            pair,
            node_cost_weight=node_cost_weight,
            beam_ms_cost_weight=beam_ms_cost_weight,
        )
        if reward is None:
            skipped_unknown += 1
            continue
        if not collector.steps:
            skipped_no_choice += 1
            continue
        records.append(
            rl_episode_record(
                seed=seed,
                sampler_seed=behavior_seed,
                report=report,
                reward=reward,
                pruner=pruner,
                steps=collector.steps,
            )
        )
        rewards.append(reward.total)

    return records, StablePrunerRLCollectionSummary(
        seeds_requested=len(normalized_seeds),
        trajectories_written=len(records),
        skipped_unknown_outcome=skipped_unknown,
        skipped_no_pruning_choice=skipped_no_choice,
        mean_reward=None if not rewards else mean(rewards),
    )


def write_trajectory_jsonl(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _require_schema(
    metadata: Mapping[str, object], field: str, expected: int, source: str
) -> None:
    if metadata.get(field) != expected:
        raise ValueError(f"{source} {field} mismatch")


def _outcome_score(outcome: str | None) -> float | None:
    if outcome is None:
        return None
    normalized = outcome.strip().lower()
    if normalized in {"victory", "win"}:
        return 1.0
    if normalized in {"defeat", "loss"}:
        return 0.0
    return None


def _derived_sampler_seed(base_seed: int, *, seed: int, index: int) -> int:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise ValueError("sampler_seed must be an integer")
    encoded = f"{base_seed}:{seed}:{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big", signed=False)


def _validate_non_negative_finite(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")


def _validate_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    if not seeds:
        raise ValueError("at least one seed is required")
    result: list[int] = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seeds must contain only integers")
        result.append(seed)
    if len(set(result)) != len(result):
        raise ValueError("seeds must be unique")
    return tuple(result)


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


def _clone_beam_config(config: BeamSearchConfig) -> BeamSearchConfig:
    return replace(
        config,
        beam_searchable_action_types=frozenset(config.beam_searchable_action_types),
        simulation_options=(
            None if config.simulation_options is None else dict(config.simulation_options)
        ),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seed_list, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument("--node-cost-weight", type=float, default=0.0)
    parser.add_argument("--beam-ms-cost-weight", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tools/output/stable_pruner_rl_trajectories.jsonl"),
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


async def _run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], StablePrunerRLCollectionSummary]:
    scenario_payload = json.loads(args.scenario.read_text(encoding="utf-8"))
    if not isinstance(scenario_payload, Mapping):
        raise ValueError("scenario JSON root must be an object")
    scenario = _scenario_from_json(scenario_payload)
    seeds = args.seeds if args.seeds is not None else (scenario.seed,)
    connection = TcpConnection(
        host=args.host,
        port=args.port,
        connect_timeout_s=args.connect_timeout,
    )
    async with AsyncTrainingApiClient(connection) as client:
        return await collect_rl_trajectories(
            client,
            scenario=scenario,
            weights=args.weights,
            seeds=seeds,
            beam_config=_cli_beam_config(args),
            temperature=args.temperature,
            sampler_seed=args.sampler_seed,
            node_cost_weight=args.node_cost_weight,
            beam_ms_cost_weight=args.beam_ms_cost_weight,
            decision_timeout_s=args.decision_timeout,
            max_decisions=args.max_decisions,
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging(args.log_level)
    records, summary = asyncio.run(_run(args))
    write_trajectory_jsonl(args.output, records)
    print(json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True))
    print(f"Wrote {len(records)} trajectories to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
