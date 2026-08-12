"""Build supervised stable-pruner ranking examples from Oracle JSONL records."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sts2_training.decision.pruner_features import stable_pruner_feature_matrix
from sts2_training.decision.stable_pruner import StablePruneContext


@dataclass(frozen=True)
class TracePrunerNode:
    node_id: str
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


@dataclass(frozen=True)
class PrunerNodeTrainingExample:
    source_path: str
    decision_point_id: str
    search_id: str
    prune_step_id: str
    node_id: str
    features: tuple[float, ...]
    target_value: float
    target_source: str
    target_weight: float
    target_beam_width: int
    baseline_would_keep: bool
    oracle_kept: bool


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
) -> list[PrunerFrontierTrainingExample]:
    if terminal_weight <= 0.0 or bootstrap_weight <= 0.0:
        raise ValueError("target source weights must be positive")

    frontiers: list[PrunerFrontierTrainingExample] = []
    for path_value in paths:
        path = Path(path_value)
        for record in _iter_records(path):
            targets = _stable_target_index(record)
            decision_point_id = _string(record.get("decision_point_id"), "decision_point_id")
            for event in _sequence(record.get("search_trace")):
                if not isinstance(event, Mapping) or event.get("event_type") != "stable_prune":
                    continue
                search_id = _string(event.get("search_id"), "search_trace.search_id")
                prune_step_id = _string(
                    event.get("prune_step_id"), "search_trace.prune_step_id"
                )
                raw_nodes = _sequence(event.get("nodes"))
                nodes = tuple(_trace_node(raw) for raw in raw_nodes if isinstance(raw, Mapping))
                step_targets = {
                    node.node_id: target
                    for node in nodes
                    for target in [targets.get((prune_step_id, node.node_id))]
                    if target is not None
                }
                target_widths = {
                    _positive_int(target.get("target_beam_width"), "oracle target_beam_width")
                    for target in step_targets.values()
                }
                if not target_widths:
                    continue
                if len(target_widths) != 1:
                    raise ValueError(
                        f"mixed target_beam_width values in prune_step_id={prune_step_id!r}"
                    )
                target_beam_width = next(iter(target_widths))

                # `event.k` is the wide Oracle beam. Features describe the decision the
                # student must make, so expose the cheaper target K instead.
                context = StablePruneContext(
                    search_id=search_id,
                    prune_step_id=prune_step_id,
                    phase=str(event.get("phase") or "stable_frontier"),
                    beam_width=target_beam_width,
                    max_depth=_positive_int(event.get("max_depth"), "search_trace.max_depth"),
                    depths_completed=_non_negative_int(
                        event.get("depths_completed"), "search_trace.depths_completed"
                    ),
                    remaining_time_ms=_optional_float(event.get("remaining_time_ms")),
                )
                rows = stable_pruner_feature_matrix(nodes, context=context)
                examples: list[PrunerNodeTrainingExample] = []
                for node, features in zip(nodes, rows, strict=True):
                    target = step_targets.get(node.node_id)
                    if target is None:
                        continue
                    target_value = target.get("target_value")
                    if isinstance(target_value, bool) or not isinstance(target_value, (int, float)):
                        continue
                    target_source = str(target.get("target_source") or "no_target")
                    weight = _target_source_weight(
                        target_source,
                        terminal_weight=terminal_weight,
                        bootstrap_weight=bootstrap_weight,
                    )
                    if weight is None:
                        continue
                    examples.append(
                        PrunerNodeTrainingExample(
                            source_path=str(path),
                            decision_point_id=decision_point_id,
                            search_id=search_id,
                            prune_step_id=prune_step_id,
                            node_id=node.node_id,
                            features=features,
                            target_value=float(target_value),
                            target_source=target_source,
                            target_weight=weight,
                            target_beam_width=target_beam_width,
                            baseline_would_keep=bool(target.get("baseline_would_keep")),
                            oracle_kept=bool(target.get("oracle_kept")),
                        )
                    )
                if len(examples) >= 2:
                    frontiers.append(
                        PrunerFrontierTrainingExample(
                            source_path=str(path),
                            decision_point_id=decision_point_id,
                            search_id=search_id,
                            prune_step_id=prune_step_id,
                            target_beam_width=target_beam_width,
                            nodes=tuple(examples),
                        )
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
        nodes = frontier.nodes
        for left_index in range(len(nodes)):
            for right_index in range(left_index + 1, len(nodes)):
                left = nodes[left_index]
                right = nodes[right_index]
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
        return {}
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for target in _sequence(oracle_targets.get("stable_nodes")):
        if not isinstance(target, Mapping):
            continue
        prune_step_id = target.get("prune_step_id")
        node_id = target.get("node_id")
        if isinstance(prune_step_id, str) and isinstance(node_id, str):
            result[(prune_step_id, node_id)] = target
    return result


def _trace_node(raw: Mapping[str, Any]) -> TracePrunerNode:
    return TracePrunerNode(
        node_id=_string(raw.get("node_id"), "stable_prune.nodes[].node_id"),
        value=_float(raw.get("value"), "stable_prune.nodes[].value"),
        root_action_id=_optional_string(raw.get("root_action_id")),
        depth=_non_negative_int(raw.get("depth"), "stable_prune.nodes[].depth"),
        combat_depth=_non_negative_int(
            raw.get("combat_depth"), "stable_prune.nodes[].combat_depth"
        ),
        continuation_steps=_non_negative_int(
            raw.get("continuation_steps"), "stable_prune.nodes[].continuation_steps"
        ),
        terminal=bool(raw.get("terminal")),
        action_type=_optional_string(raw.get("action_type")),
        policy_rank=_optional_int(raw.get("policy_rank")),
        policy_score=_optional_float(raw.get("policy_score")),
        post_coverage_rank=_optional_int(raw.get("post_coverage_rank")),
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
    return value if isinstance(value, str) else None


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


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value