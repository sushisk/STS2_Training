"""Feature extraction for supervised stable-frontier pruning.

The initial learned pruner intentionally uses only information already present in the
Phase-1 stable-prune trace contract. That keeps offline training and runtime inference on
the same feature schema without requiring emulator replay or hidden state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from sts2_training.decision.stable_pruner import StablePruneContext


class PrunerFeatureNode(Protocol):
    value: float
    root_action_id: str | None
    depth: int
    combat_depth: int
    continuation_steps: int
    terminal: bool
    action_type: str | None
    policy_rank: int | None
    policy_score: float | None
    post_coverage_rank: int | None
    candidate_source: str | None


PRUNER_FEATURE_SCHEMA_VERSION = 1
PRUNER_FEATURE_NAMES = (
    "node_value",
    "frontier_value_max",
    "frontier_value_min",
    "frontier_value_mean",
    "frontier_value_std",
    "value_gap_to_max",
    "value_zscore",
    "value_rank_fraction",
    "root_group_size",
    "root_group_fraction",
    "root_group_value_max",
    "root_group_value_mean",
    "root_group_value_gap_to_max",
    "within_root_value_rank_fraction",
    "depth",
    "combat_depth",
    "continuation_steps",
    "terminal",
    "policy_rank_missing",
    "policy_rank",
    "policy_score_missing",
    "policy_score",
    "post_coverage_rank_missing",
    "post_coverage_rank",
    "candidate_source_structural_coverage",
    "action_type_card",
    "action_type_system",
    "action_type_potion",
    "remaining_depth",
    "beam_width",
    "frontier_size",
    "remaining_time_ms_missing",
    "remaining_time_ms_log1p",
)


def stable_pruner_feature_matrix(
    frontier: Sequence[PrunerFeatureNode],
    *,
    context: StablePruneContext,
) -> list[tuple[float, ...]]:
    if not frontier:
        return []

    values = [float(node.value) for node in frontier]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("stable pruner node values must be finite")
    value_max = max(values)
    value_min = min(values)
    value_mean = sum(values) / len(values)
    variance = sum((value - value_mean) ** 2 for value in values) / len(values)
    value_std = math.sqrt(max(0.0, variance))

    global_rank = _stable_descending_rank(values)
    groups: dict[str | None, list[int]] = {}
    for index, node in enumerate(frontier):
        groups.setdefault(node.root_action_id, []).append(index)

    group_stats: dict[str | None, tuple[float, float, dict[int, int]]] = {}
    for root_action_id, indices in groups.items():
        group_values = [values[index] for index in indices]
        group_max = max(group_values)
        group_mean = sum(group_values) / len(group_values)
        local_ranks = _stable_descending_rank(group_values)
        group_stats[root_action_id] = (
            group_max,
            group_mean,
            {index: local_ranks[position] for position, index in enumerate(indices)},
        )

    remaining_time = context.remaining_time_ms
    if remaining_time is not None and not math.isfinite(float(remaining_time)):
        remaining_time = None
    remaining_time_missing = 1.0 if remaining_time is None else 0.0
    remaining_time_log1p = 0.0 if remaining_time is None else math.log1p(max(0.0, remaining_time))
    remaining_depth = max(0, context.max_depth - context.depths_completed)

    rows: list[tuple[float, ...]] = []
    for index, node in enumerate(frontier):
        group_indices = groups[node.root_action_id]
        group_max, group_mean, local_ranks = group_stats[node.root_action_id]
        group_size = len(group_indices)
        policy_score = _finite_optional(node.policy_score)
        rows.append(
            (
                values[index],
                value_max,
                value_min,
                value_mean,
                value_std,
                value_max - values[index],
                0.0 if value_std == 0.0 else (values[index] - value_mean) / value_std,
                _rank_fraction(global_rank[index], len(frontier)),
                float(group_size),
                group_size / len(frontier),
                group_max,
                group_mean,
                group_max - values[index],
                _rank_fraction(local_ranks[index], group_size),
                float(node.depth),
                float(node.combat_depth),
                float(node.continuation_steps),
                1.0 if node.terminal else 0.0,
                1.0 if node.policy_rank is None else 0.0,
                0.0 if node.policy_rank is None else float(node.policy_rank),
                1.0 if policy_score is None else 0.0,
                0.0 if policy_score is None else policy_score,
                1.0 if node.post_coverage_rank is None else 0.0,
                0.0 if node.post_coverage_rank is None else float(node.post_coverage_rank),
                1.0 if node.candidate_source == "structural_coverage" else 0.0,
                1.0 if node.action_type == "card" else 0.0,
                1.0 if node.action_type == "system" else 0.0,
                1.0 if node.action_type == "potion" else 0.0,
                float(remaining_depth),
                float(context.beam_width),
                float(len(frontier)),
                remaining_time_missing,
                remaining_time_log1p,
            )
        )
    return rows


def _stable_descending_rank(values: Sequence[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    ranks = [0] * len(values)
    for rank, index in enumerate(order):
        ranks[index] = rank
    return ranks


def _rank_fraction(rank: int, size: int) -> float:
    if size <= 1:
        return 0.0
    return rank / (size - 1)


def _finite_optional(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None
