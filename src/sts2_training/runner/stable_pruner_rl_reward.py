"""Versioned reward construction for stable-pruner RL."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from sts2_training.runner.combat_resource_reward import (
    COMBAT_RESOURCE_EVALUATOR_VERSION,
    COMBAT_RESOURCE_HP_WEIGHT,
    COMBAT_RESOURCE_POTION_WEIGHT,
    COMBAT_RESOURCE_REWARD_WEIGHT,
    CombatResourceSnapshot,
    combat_resource_quality,
)

RL_TRAJECTORY_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class PairedPrunerReward:
    outcome_delta: float
    resource_evaluator_version: int
    resource_hp_weight: float
    resource_potion_weight: float
    resource_quality_delta: float
    resource_reward_weight: float
    nodes_expanded_delta: int
    beam_ms_delta: float
    node_cost_weight: float
    beam_ms_cost_weight: float
    total: float


def paired_pruner_reward(
    pair: Any,
    *,
    node_cost_weight: float = 0.0,
    beam_ms_cost_weight: float = 0.0,
) -> PairedPrunerReward | None:
    _non_negative("node_cost_weight", node_cost_weight)
    _non_negative("beam_ms_cost_weight", beam_ms_cost_weight)
    baseline_outcome = _outcome_score(pair.baseline.outcome)
    learned_outcome = _outcome_score(pair.learned.outcome)
    if baseline_outcome is None or learned_outcome is None:
        return None
    baseline_quality = arm_resource_quality(pair.baseline)
    learned_quality = arm_resource_quality(pair.learned)
    outcome_delta = learned_outcome - baseline_outcome
    resource_delta = learned_quality - baseline_quality
    nodes_delta = pair.learned.nodes_expanded - pair.baseline.nodes_expanded
    beam_ms_delta = pair.learned.beam_total_ms - pair.baseline.beam_total_ms
    total = (
        outcome_delta
        + COMBAT_RESOURCE_REWARD_WEIGHT * resource_delta
        - float(node_cost_weight) * nodes_delta
        - float(beam_ms_cost_weight) * beam_ms_delta
    )
    return PairedPrunerReward(
        outcome_delta=outcome_delta,
        resource_evaluator_version=COMBAT_RESOURCE_EVALUATOR_VERSION,
        resource_hp_weight=COMBAT_RESOURCE_HP_WEIGHT,
        resource_potion_weight=COMBAT_RESOURCE_POTION_WEIGHT,
        resource_quality_delta=resource_delta,
        resource_reward_weight=COMBAT_RESOURCE_REWARD_WEIGHT,
        nodes_expanded_delta=nodes_delta,
        beam_ms_delta=beam_ms_delta,
        node_cost_weight=float(node_cost_weight),
        beam_ms_cost_weight=float(beam_ms_cost_weight),
        total=total,
    )


def paired_result_record(pair: Any) -> dict[str, Any]:
    return {
        "baseline_outcome": pair.baseline.outcome,
        "learned_outcome": pair.learned.outcome,
        "winner": pair.winner,
        "baseline_terminal_hp": pair.baseline.terminal_hp,
        "learned_terminal_hp": pair.learned.terminal_hp,
        "baseline_terminal_max_hp": pair.baseline.terminal_max_hp,
        "learned_terminal_max_hp": pair.learned.terminal_max_hp,
        "baseline_terminal_potion_count": pair.baseline.terminal_potion_count,
        "learned_terminal_potion_count": pair.learned.terminal_potion_count,
        "baseline_initial_potion_count": pair.baseline.initial_potion_count,
        "learned_initial_potion_count": pair.learned.initial_potion_count,
        "baseline_resource_quality": arm_resource_quality(pair.baseline),
        "learned_resource_quality": arm_resource_quality(pair.learned),
        "baseline_nodes_expanded": pair.baseline.nodes_expanded,
        "learned_nodes_expanded": pair.learned.nodes_expanded,
        "baseline_beam_total_ms": pair.baseline.beam_total_ms,
        "learned_beam_total_ms": pair.learned.beam_total_ms,
        "common_action_prefix": pair.common_action_prefix,
        "first_divergence_index": pair.first_divergence_index,
    }


def arm_resource_quality(arm: Any) -> float:
    score = combat_resource_quality(
        CombatResourceSnapshot(
            hp=_finite(arm.terminal_hp, "terminal_hp"),
            max_hp=_finite(arm.terminal_max_hp, "terminal_max_hp"),
            potion_count=_integer(arm.terminal_potion_count, "terminal_potion_count"),
            initial_potion_count=_integer(arm.initial_potion_count, "initial_potion_count"),
        )
    )
    logged = getattr(arm, "resource_quality", None)
    if logged is not None and not math.isclose(_finite(logged, "resource_quality"), score, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("resource_quality does not match frozen evaluator")
    return score


def _outcome_score(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return 1.0 if value in {"victory", "win"} else 0.0 if value in {"defeat", "loss"} else None


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite numeric")
    return float(value)


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _non_negative(name: str, value: float) -> None:
    if _finite(value, name) < 0:
        raise ValueError(f"{name} must be non-negative")
