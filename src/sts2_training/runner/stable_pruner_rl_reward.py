"""Versioned resource-aware reward for stable-pruner RL."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any
from sts2_training.runner.combat_resource_reward import (
    COMBAT_RESOURCE_EVALUATOR_VERSION, COMBAT_RESOURCE_HP_WEIGHT,
    COMBAT_RESOURCE_POTION_WEIGHT, COMBAT_RESOURCE_REWARD_WEIGHT,
    CombatResourceSnapshot, combat_resource_quality,
)
RL_TRAJECTORY_SCHEMA_VERSION = 3

@dataclass(frozen=True)
class PairedPrunerReward:
    outcome_delta: float
    nodes_expanded_delta: int
    beam_ms_delta: float
    node_cost_weight: float
    beam_ms_cost_weight: float
    total: float
    resource_evaluator_version: int
    resource_hp_weight: float
    resource_potion_weight: float
    resource_reward_weight: float
    resource_quality_delta: float
    baseline_terminal_hp: float
    baseline_terminal_max_hp: float
    baseline_terminal_potion_count: int
    baseline_initial_potion_count: int
    baseline_resource_quality: float
    learned_terminal_hp: float
    learned_terminal_max_hp: float
    learned_terminal_potion_count: int
    learned_initial_potion_count: int
    learned_resource_quality: float

def paired_pruner_reward(pair: Any, *, node_cost_weight: float = 0.0, beam_ms_cost_weight: float = 0.0) -> PairedPrunerReward | None:
    _non_negative(node_cost_weight); _non_negative(beam_ms_cost_weight)
    bo, lo = _outcome(pair.baseline.outcome), _outcome(pair.learned.outcome)
    if bo is None or lo is None:
        return None
    bq, lq = arm_resource_quality(pair.baseline), arm_resource_quality(pair.learned)
    od, rd = lo - bo, lq - bq
    nd = pair.learned.nodes_expanded - pair.baseline.nodes_expanded
    bd = pair.learned.beam_total_ms - pair.baseline.beam_total_ms
    total = od + COMBAT_RESOURCE_REWARD_WEIGHT * rd - node_cost_weight * nd - beam_ms_cost_weight * bd
    b, l = pair.baseline, pair.learned
    return PairedPrunerReward(
        od, nd, bd, float(node_cost_weight), float(beam_ms_cost_weight), total,
        COMBAT_RESOURCE_EVALUATOR_VERSION, COMBAT_RESOURCE_HP_WEIGHT,
        COMBAT_RESOURCE_POTION_WEIGHT, COMBAT_RESOURCE_REWARD_WEIGHT, rd,
        _finite(b.terminal_hp), _finite(b.terminal_max_hp), _integer(b.terminal_potion_count),
        _integer(b.initial_potion_count), bq, _finite(l.terminal_hp), _finite(l.terminal_max_hp),
        _integer(l.terminal_potion_count), _integer(l.initial_potion_count), lq)

def arm_resource_quality(arm: Any) -> float:
    return combat_resource_quality(CombatResourceSnapshot(
        hp=_finite(arm.terminal_hp), max_hp=_finite(arm.terminal_max_hp),
        potion_count=_integer(arm.terminal_potion_count),
        initial_potion_count=_integer(arm.initial_potion_count)))

def _outcome(value: Any) -> float | None:
    if not isinstance(value, str): return None
    value = value.strip().lower()
    return 1.0 if value in {"victory", "win"} else 0.0 if value in {"defeat", "loss"} else None

def _finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("reward resource field must be finite numeric")
    return float(value)

def _integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int): raise ValueError("reward resource field must be integer")
    return value

def _non_negative(value: float) -> None:
    if _finite(value) < 0: raise ValueError("reward cost weight must be non-negative")
