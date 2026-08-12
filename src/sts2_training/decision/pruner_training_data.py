"""Build supervised stable-pruner ranking examples from Oracle JSONL records."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sts2_training.decision.oracle_teacher_provenance import (
    inspect_oracle_teacher_provenance,
)
from sts2_training.decision.pruner_features import (
    PRUNER_FEATURE_NAMES,
    stable_pruner_feature_matrix,
)
from sts2_training.decision.search_trace import StablePruneNodeTrace, StablePruneTrace


@dataclass(frozen=True)
class PrunerNodeTrainingExample:
    source_path: str
    decision_point_id: str
    search_id: str
    prune_step_id: str
    node_id: str
    features: tuple[float, ...]
    target_value: float | None
    target_source: str
    target_weight: float | None
    target_beam_width: int
    baseline_would_keep: bool
    oracle_kept: bool
    censored: bool = False
    censor_reason: str | None = None


@dataclass(frozen=True)
class PrunerFrontierTrainingExample:
    source_path: str
    decision_point_id: str
    search_id: str
    prune_step_id: str
    target_beam_width: int
    nodes: tuple[PrunerNodeTrainingExample, ...]


@dataclass(frozen=True)
class PairwisePrunerExample:
    features: tuple[float, ...]
    label: int
    weight: float
    prune_step_id: str
    positive_node_id: str
    negative_node_id: str
    target_gap: float


def load_pruner_frontiers(
    paths: Iterable[str | Path],
    *,
    terminal_weight: float = 1.0,
    bootstrap_weight: float = 0.5,
    allow_mixed_teachers: bool = False,
) -> list[PrunerFrontierTrainingExample]:
    if terminal_weight <= 0.0 or bootstrap_weight <= 0.0:
        raise ValueError("target source weights must be positive")

    normalized_paths = tuple(Path(path) for path in paths)
    inspect_oracle_teacher_provenance(
        normalized_paths,
        allow_mixed_teachers=allow_mixed_teachers,
    )

    frontiers: list[PrunerFrontierTrainingExample] = []
    for path in normalized_paths:
        for record in _iter_records(path):
            targets = _stable_target_index(record)
            decision_point_id = _string(record.get("decision_point_id"), "decision_point_id")
            trace_target_keys: set[tuple[str, str]] = set()
            for event in _sequence(record.get("search_trace")):
                if not isinstance(event, Mapping) or event.get("event_type") != "stable_prune":
                    continue

                # Deserialize trace identity once, then use only the upstream replay helpers
                # for learned feature input. This keeps runtime and offline featurization on
                # exactly StablePruneNodeView + StablePruneContext.
                trace = _stable_prune_trace(event)
                trace_keys = {
                    (trace.prune_step_id, node.node_id)
                    for node in trace.nodes
                }
                step_target_keys = {
                    key for key in targets if key[0] == trace.prune_step_id
                }
                missing = trace_keys - step_target_keys
                extra = step_target_keys - trace_keys
                if missing or extra:
                    raise ValueError(
                        f"{path}: stable prune target mismatch for "
                        f"prune_step_id={trace.prune_step_id!r}: "
                        f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
                    )
                trace_target_keys.update(trace_keys)
                step_targets = {
                    node.node_id: targets[(trace.prune_step_id, node.node_id)]
                    for node in trace.nodes
                }
                if not step_targets:
                    continue
                target_widths = {
                    _positive_int(target.get("target_beam_width"), "oracle target_beam_width")
                    for target in step_targets.values()
                }
                if len(target_widths) != 1:
                    raise ValueError(
                        f"mixed target_beam_width values in prune_step_id={trace.prune_step_id!r}"
                    )
                target_beam_width = next(iter(target_widths))

                # Oracle v3 records the wide teacher depth/time budget. Only target K is a
                # reconstructible student-budget field, so feature schema v2 uses this
                # context for beam_width while excluding depth/time budget-derived features.
                views = trace.node_views()
                context = trace.to_prune_context(beam_width=target_beam_width)
                rows = stable_pruner_feature_matrix(views, context=context)

                examples: list[PrunerNodeTrainingExample] = []
                for node, features in zip(trace.nodes, rows, strict=True):
                    target = step_targets[node.node_id]
                    raw_target_value = target.get("target_value")
                    numeric_target = (
                        not isinstance(raw_target_value, bool)
                        and isinstance(raw_target_value, (int, float))
                    )
                    target_source = str(target.get("target_source") or "no_target")
                    weight = (
                        _target_source_weight(
                            target_source,
                            terminal_weight=terminal_weight,
                            bootstrap_weight=bootstrap_weight,
                        )
                        if numeric_target
                        else None
                    )
                    # A node remains in the frontier even when its downstream target is
                    # censored/unobserved. Pairwise construction filters these later, while
                    # selection evaluation must rank the same full frontier seen at runtime.
                    target_value = float(raw_target_value) if weight is not None else None
                    examples.append(
                        PrunerNodeTrainingExample(
                            source_path=str(path),
                            decision_point_id=decision_point_id,
                            search_id=trace.search_id,
                            prune_step_id=trace.prune_step_id,
                            node_id=node.node_id,
                            features=features,
                            target_value=target_value,
                            target_source=target_source,
                            target_weight=weight,
                            target_beam_width=target_beam_width,
                            baseline_would_keep=bool(target.get("baseline_would_keep")),
                            oracle_kept=bool(target.get("oracle_kept")),
                            censored=bool(target.get("censored")),
                            censor_reason=_optional_string(target.get("censor_reason")),
                        )
                    )
                if examples:
                    frontiers.append(
                        PrunerFrontierTrainingExample(
                            source_path=str(path),
                            decision_point_id=decision_point_id,
                            search_id=trace.search_id,
                            prune_step_id=trace.prune_step_id,
                            target_beam_width=target_beam_width,
                            nodes=tuple(examples),
                        )
                    )

            unmatched_targets = set(targets) - trace_target_keys
            if unmatched_targets:
                raise ValueError(
                    f"{path}: Oracle stable targets have no matching stable_prune trace nodes: "
                    f"{sorted(unmatched_targets)!r}"
                )
    return frontiers


def build_pairwise_examples(
    frontiers: Sequence[PrunerFrontierTrainingExample],
    *,
    min_target_gap: float = 1e-6,
) -> list[PairwisePrunerExample]:
    if min_target_gap < 0.0:
        raise ValueError("min_target_gap must be non-negative")

    pairs: list[PairwisePrunerExample] = []
    for frontier in frontiers:
        nodes = [
            node
            for node in frontier.nodes
            if node.target_value is not None and node.target_weight is not None
        ]
        for left_index in range(len(nodes)):
            for right_index in range(left_index + 1, len(nodes)):
                left = nodes[left_index]
                right = nodes[right_index]
                assert left.target_value is not None and right.target_value is not None
                assert left.target_weight is not None and right.target_weight is not None
                gap = left.target_value - right.target_value
                if abs(gap) <= min_target_gap:
                    continue
                better, worse = (left, right) if gap > 0 else (right, left)
                difference = tuple(
                    better_value - worse_value
                    for better_value, worse_value in zip(
                        better.features,
                        worse.features,
                        strict=True,
                    )
                )
                weight = min(better.target_weight, worse.target_weight)
                target_gap = abs(gap)
                pairs.append(
                    PairwisePrunerExample(
                        features=difference,
                        label=1,
                        weight=weight,
                        prune_step_id=frontier.prune_step_id,
                        positive_node_id=better.node_id,
                        negative_node_id=worse.node_id,
                        target_gap=target_gap,
                    )
                )
                pairs.append(
                    PairwisePrunerExample(
                        features=tuple(-value for value in difference),
                        label=0,
                        weight=weight,
                        prune_step_id=frontier.prune_step_id,
                        positive_node_id=worse.node_id,
                        negative_node_id=better.node_id,
                        target_gap=target_gap,
                    )
                )
    return pairs


def evaluate_selection_metrics(
    frontiers: Sequence[PrunerFrontierTrainingExample],
    *,
    score_node: Callable[[PrunerNodeTrainingExample], float],
) -> dict[str, Any]:
    """Evaluate runtime-like top-K selection without inventing targets for censored nodes.

    Every frontier node participates in learned/baseline ranking and consumes K exactly as
    it can at runtime. Recall is measured against the best labeled nodes and therefore drops
    when a selected ``no_target`` node displaces a known-good node. Regret/mean-gap metrics
    are reported only for frontiers whose selected K and teacher top-K are fully labeled.
    Coverage and selected-no-target rates make the censoring boundary explicit.
    """

    value_feature_index = PRUNER_FEATURE_NAMES.index("node_value")
    learned_recall: list[float] = []
    baseline_recall: list[float] = []
    learned_regret: list[float] = []
    baseline_regret: list[float] = []
    learned_mean_gap: list[float] = []
    baseline_mean_gap: list[float] = []
    frontier_coverages: list[float] = []
    total_nodes = 0
    labeled_nodes = 0
    learned_selected = 0
    baseline_selected = 0
    learned_selected_no_target = 0
    baseline_selected_no_target = 0
    recall_evaluable_frontiers = 0
    learned_quality_evaluable_frontiers = 0
    baseline_quality_evaluable_frontiers = 0

    for frontier in frontiers:
        nodes = list(frontier.nodes)
        if not nodes:
            continue
        labeled = [node for node in nodes if node.target_value is not None]
        total_nodes += len(nodes)
        labeled_nodes += len(labeled)
        frontier_coverages.append(len(labeled) / len(nodes))

        k = min(frontier.target_beam_width, len(nodes))
        learned = sorted(nodes, key=score_node, reverse=True)[:k]
        baseline = sorted(
            nodes,
            key=lambda node: node.features[value_feature_index],
            reverse=True,
        )[:k]
        learned_selected += len(learned)
        baseline_selected += len(baseline)
        learned_selected_no_target += sum(node.target_value is None for node in learned)
        baseline_selected_no_target += sum(node.target_value is None for node in baseline)

        teacher_count = min(k, len(labeled))
        if teacher_count:
            teacher = sorted(labeled, key=_target_value, reverse=True)[:teacher_count]
            teacher_ids = {node.node_id for node in teacher}
            learned_recall.append(
                len(teacher_ids & {node.node_id for node in learned}) / teacher_count
            )
            baseline_recall.append(
                len(teacher_ids & {node.node_id for node in baseline}) / teacher_count
            )
            recall_evaluable_frontiers += 1

        if len(labeled) < k:
            continue
        teacher = sorted(labeled, key=_target_value, reverse=True)[:k]
        best_target = max(_target_value(node) for node in labeled)
        teacher_mean = sum(_target_value(node) for node in teacher) / k

        if all(node.target_value is not None for node in learned):
            learned_regret.append(
                best_target - max(_target_value(node) for node in learned)
            )
            learned_mean_gap.append(
                teacher_mean - sum(_target_value(node) for node in learned) / k
            )
            learned_quality_evaluable_frontiers += 1
        if all(node.target_value is not None for node in baseline):
            baseline_regret.append(
                best_target - max(_target_value(node) for node in baseline)
            )
            baseline_mean_gap.append(
                teacher_mean - sum(_target_value(node) for node in baseline) / k
            )
            baseline_quality_evaluable_frontiers += 1

    return {
        "frontiers": len(frontiers),
        "nodes": total_nodes,
        "labeled_nodes": labeled_nodes,
        "label_coverage": None if total_nodes == 0 else labeled_nodes / total_nodes,
        "mean_frontier_label_coverage": _mean_or_none(frontier_coverages),
        "recall_evaluable_frontiers": recall_evaluable_frontiers,
        "learned_selected_no_target_rate": (
            None
            if learned_selected == 0
            else learned_selected_no_target / learned_selected
        ),
        "value_top_k_selected_no_target_rate": (
            None
            if baseline_selected == 0
            else baseline_selected_no_target / baseline_selected
        ),
        "learned_recall_at_k": _mean_or_none(learned_recall),
        "value_top_k_recall_at_k": _mean_or_none(baseline_recall),
        "learned_quality_evaluable_frontiers": learned_quality_evaluable_frontiers,
        "value_top_k_quality_evaluable_frontiers": baseline_quality_evaluable_frontiers,
        "learned_best_value_regret": _mean_or_none(learned_regret),
        "value_top_k_best_value_regret": _mean_or_none(baseline_regret),
        "learned_mean_selected_value_gap": _mean_or_none(learned_mean_gap),
        "value_top_k_mean_selected_value_gap": _mean_or_none(baseline_mean_gap),
    }


def _target_value(node: PrunerNodeTrainingExample) -> float:
    if node.target_value is None:
        raise ValueError("target_value is unavailable for censored node")
    return node.target_value


def _mean_or_none(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _iter_records(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: oracle JSONL record must be an object")
            if value.get("record_type") != "combat_oracle_decision":
                continue
            yield value


def _stable_target_index(record: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    oracle_targets = record.get("oracle_targets")
    if not isinstance(oracle_targets, Mapping):
        raise ValueError("oracle_targets must be an object")
    raw_targets = oracle_targets.get("stable_nodes")
    if not isinstance(raw_targets, Sequence) or isinstance(
        raw_targets, (str, bytes, bytearray)
    ):
        raise ValueError("oracle_targets.stable_nodes must be a sequence")

    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, target in enumerate(raw_targets):
        if not isinstance(target, Mapping):
            raise ValueError(f"oracle_targets.stable_nodes[{index}] must be an object")
        prune_step_id = _string(
            target.get("prune_step_id"),
            f"oracle_targets.stable_nodes[{index}].prune_step_id",
        )
        node_id = _string(
            target.get("node_id"),
            f"oracle_targets.stable_nodes[{index}].node_id",
        )
        key = (prune_step_id, node_id)
        if key in result:
            raise ValueError(f"duplicate Oracle stable target for key={key!r}")
        result[key] = target
    return result


def _stable_prune_trace(raw: Mapping[str, Any]) -> StablePruneTrace:
    nodes = tuple(
        _stable_prune_node_trace(node)
        for node in _sequence(raw.get("nodes"))
        if isinstance(node, Mapping)
    )
    return StablePruneTrace(
        search_id=_string(raw.get("search_id"), "search_trace.search_id"),
        prune_step_id=_string(raw.get("prune_step_id"), "search_trace.prune_step_id"),
        phase=_string(raw.get("phase"), "search_trace.phase"),
        k=_positive_int(raw.get("k"), "search_trace.k"),
        frontier_size=_non_negative_int(raw.get("frontier_size"), "search_trace.frontier_size"),
        pruner_name=_string(raw.get("pruner_name"), "search_trace.pruner_name"),
        pruner_version=_string(raw.get("pruner_version"), "search_trace.pruner_version"),
        max_depth=_positive_int(raw.get("max_depth"), "search_trace.max_depth"),
        depths_completed=_non_negative_int(
            raw.get("depths_completed"), "search_trace.depths_completed"
        ),
        remaining_time_ms=_optional_float(raw.get("remaining_time_ms")),
        nodes=nodes,
    )


def _stable_prune_node_trace(raw: Mapping[str, Any]) -> StablePruneNodeTrace:
    return StablePruneNodeTrace(
        node_id=_string(raw.get("node_id"), "stable_prune.nodes[].node_id"),
        parent_node_id=_string(
            raw.get("parent_node_id"), "stable_prune.nodes[].parent_node_id"
        ),
        branch_id=_string(raw.get("branch_id"), "stable_prune.nodes[].branch_id"),
        parent_branch_id=_string(
            raw.get("parent_branch_id"), "stable_prune.nodes[].parent_branch_id"
        ),
        frontier_index_before_prune=_non_negative_int(
            raw.get("frontier_index_before_prune"),
            "stable_prune.nodes[].frontier_index_before_prune",
        ),
        kept=_bool(raw.get("kept"), "stable_prune.nodes[].kept"),
        value=_float(raw.get("value"), "stable_prune.nodes[].value"),
        root_action_id=_optional_string(raw.get("root_action_id")),
        rng_id=_int(raw.get("rng_id"), "stable_prune.nodes[].rng_id"),
        decision_point_id=_string(
            raw.get("decision_point_id"), "stable_prune.nodes[].decision_point_id"
        ),
        depth=_non_negative_int(raw.get("depth"), "stable_prune.nodes[].depth"),
        combat_depth=_non_negative_int(
            raw.get("combat_depth"), "stable_prune.nodes[].combat_depth"
        ),
        continuation_steps=_non_negative_int(
            raw.get("continuation_steps"), "stable_prune.nodes[].continuation_steps"
        ),
        terminal=_bool(raw.get("terminal"), "stable_prune.nodes[].terminal"),
        action_id=_optional_string(raw.get("action_id")),
        action_type=_optional_string(raw.get("action_type")),
        action=_optional_mapping(raw.get("action")),
        policy_rank=_optional_non_negative_int(raw.get("policy_rank")),
        policy_score=_optional_float(raw.get("policy_score")),
        post_coverage_rank=_optional_non_negative_int(raw.get("post_coverage_rank")),
        candidate_source=_optional_string(raw.get("candidate_source")),
    )


def _target_source_weight(
    target_source: str,
    *,
    terminal_weight: float,
    bootstrap_weight: float,
) -> float | None:
    if target_source == "terminal":
        return terminal_weight
    if target_source == "value_bootstrap":
        return bootstrap_weight
    return None


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _optional_non_negative_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a bool")
    return value


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None
