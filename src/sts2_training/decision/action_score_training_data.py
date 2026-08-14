"""Load supervised Combat ``action_score`` rankings from Oracle v6 JSONL records."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from sts2_training.decision.action_score_features import combat_action_score_features
from sts2_training.decision.oracle_log import (
    ORACLE_RECORD_SCHEMA_VERSION,
    require_oracle_value_dto_version,
)
from sts2_training.decision.oracle_teacher_provenance import inspect_oracle_teacher_provenance
from sts2_training.decision.value_training_data import inspect_oracle_value_dto_contract


@dataclass(frozen=True)
class CombatActionScoreTrainingExample:
    source_path: str
    instance_id: str
    decision_index: int
    decision_point_id: str
    action_id: str
    action: Mapping[str, Any]
    features: tuple[float, ...]
    estimated_q: float
    target_source: str
    sample_weight: float
    censored: bool
    censor_reason: str | None
    terminal_reached: bool
    dto_version: str

    @property
    def decision_key(self) -> tuple[str, str, int]:
        return (self.source_path, self.instance_id, self.decision_index)

    @property
    def node_score(self) -> float:
        """Canonical name for the Oracle node score persisted as ``estimated_q``."""

        return self.estimated_q


@dataclass(frozen=True)
class PairwiseActionScoreExample:
    decision_key: tuple[str, str, int]
    winner_action_id: str
    loser_action_id: str
    feature_delta: tuple[float, ...]
    label: int
    sample_weight: float


@dataclass(frozen=True)
class CombatActionScoreDatasetStats:
    decision_records: int
    root_actions: int
    usable_actions: int
    no_target_actions: int
    unresolved_actions: int
    censored_actions: int
    dto_version: str | None = None

    @property
    def label_coverage(self) -> float | None:
        return None if self.root_actions == 0 else self.usable_actions / self.root_actions

    def to_json(self) -> dict[str, Any]:
        return {
            "decision_records": self.decision_records,
            "root_actions": self.root_actions,
            "usable_actions": self.usable_actions,
            "no_target_actions": self.no_target_actions,
            "unresolved_actions": self.unresolved_actions,
            "censored_actions": self.censored_actions,
            "label_coverage": self.label_coverage,
            "dto_version": self.dto_version,
        }


def load_combat_action_score_examples(
    paths: Iterable[str | Path],
    *,
    terminal_weight: float = 1.0,
    bootstrap_weight: float = 0.5,
    mixed_weight: float = 0.5,
    allow_mixed_teachers: bool = False,
    require_exhaustive_root_actions: bool = True,
) -> tuple[list[CombatActionScoreTrainingExample], CombatActionScoreDatasetStats]:
    """Load usable pre-action candidates and their Oracle ``node_score`` labels.

    Oracle v6 persists the aggregate node score under the compatibility key
    ``estimated_q``. ``no_target`` actions and actions with any unresolved RNG outcome are
    excluded rather than converting a censored aggregate into a fabricated target. Fully
    resolved value-bootstrap/mixed targets remain usable with explicit lower sample
    weights, matching the existing ValueModel treatment of approximate Oracle labels.
    """

    weights = {
        "terminal": _positive_weight(terminal_weight, "terminal_weight"),
        "value_bootstrap": _positive_weight(bootstrap_weight, "bootstrap_weight"),
        "mixed": _positive_weight(mixed_weight, "mixed_weight"),
    }
    if not isinstance(require_exhaustive_root_actions, bool):
        raise TypeError("require_exhaustive_root_actions must be a bool")
    normalized_paths = tuple(Path(path) for path in paths)
    if not normalized_paths:
        raise ValueError("Combat action_score training requires at least one Oracle JSONL path")

    contract = inspect_oracle_value_dto_contract(normalized_paths)
    inspect_oracle_teacher_provenance(
        normalized_paths,
        allow_mixed_teachers=allow_mixed_teachers,
    )

    examples: list[CombatActionScoreTrainingExample] = []
    decision_records = 0
    root_actions = 0
    no_target_actions = 0
    unresolved_actions = 0
    censored_actions = 0

    for path in normalized_paths:
        for line_number, record in _records(path):
            if record.get("record_type") != "combat_oracle_decision":
                continue
            source_prefix = f"{path}:{line_number}"
            decision_records += 1
            if record.get("record_schema_version") != ORACLE_RECORD_SCHEMA_VERSION:
                raise ValueError(
                    f"{source_prefix}: expected Oracle record schema "
                    f"v{ORACLE_RECORD_SCHEMA_VERSION}, got {record.get('record_schema_version')!r}"
                )
            instance_id = _string(record.get("instance_id"), f"{source_prefix}.instance_id")
            decision_index = _nonnegative_int(
                record.get("decision_index"), f"{source_prefix}.decision_index"
            )
            decision_point_id = _string(
                record.get("decision_point_id"), f"{source_prefix}.decision_point_id"
            )
            dto = _mapping(
                record.get("masked_emulator_dto"), f"{source_prefix}.masked_emulator_dto"
            )
            require_oracle_value_dto_version(
                dto,
                expected=contract.dto_version,
                context=f"{source_prefix}.masked_emulator_dto",
            )

            oracle_targets = _mapping(
                record.get("oracle_targets"), f"{source_prefix}.oracle_targets"
            )
            metadata = _mapping(
                oracle_targets.get("metadata"), f"{source_prefix}.oracle_targets.metadata"
            )
            exhaustive = metadata.get("exhaustive_root_actions")
            if not isinstance(exhaustive, bool):
                raise ValueError(
                    f"{source_prefix}.oracle_targets.metadata.exhaustive_root_actions must be a bool"
                )
            if require_exhaustive_root_actions and not exhaustive:
                raise ValueError(
                    f"{source_prefix}: action_score training requires exhaustive_root_actions=True "
                    "so labels are not pre-filtered by the teacher policy"
                )

            raw_actions = _sequence(
                oracle_targets.get("root_actions"),
                f"{source_prefix}.oracle_targets.root_actions",
            )
            seen_action_ids: set[str] = set()
            for action_index, raw in enumerate(raw_actions):
                source = f"{source_prefix}.oracle_targets.root_actions[{action_index}]"
                if not isinstance(raw, Mapping):
                    raise ValueError(f"{source}: root action target must be an object")
                root_actions += 1
                action_id = _string(raw.get("action_id"), f"{source}.action_id")
                if action_id in seen_action_ids:
                    raise ValueError(f"{source}: duplicate root action_id {action_id!r}")
                seen_action_ids.add(action_id)
                action = _mapping(raw.get("action"), f"{source}.action")
                if action.get("action_id") not in {None, action_id}:
                    raise ValueError(f"{source}.action.action_id does not match action_id")
                evaluated = _bool(raw.get("evaluated"), f"{source}.evaluated")
                target_source = _string(raw.get("target_source"), f"{source}.target_source")
                raw_node_score = raw.get("estimated_q")
                censored = _bool(raw.get("censored"), f"{source}.censored")
                if censored:
                    censored_actions += 1
                censor_reason = _optional_string(
                    raw.get("censor_reason"), f"{source}.censor_reason"
                )
                terminal_reached = _bool(
                    raw.get("terminal_reached"), f"{source}.terminal_reached"
                )

                if target_source == "no_target":
                    if raw_node_score is not None:
                        raise ValueError(f"{source}: no_target action must have null estimated_q")
                    no_target_actions += 1
                    continue
                if target_source not in weights:
                    raise ValueError(f"{source}: unsupported target_source {target_source!r}")
                if not evaluated:
                    raise ValueError(f"{source}: labeled action must have evaluated=true")
                node_score = _finite_float(raw_node_score, f"{source}.estimated_q")
                if not _all_rng_outcomes_resolved(raw.get("rng_outcomes"), source=source):
                    unresolved_actions += 1
                    continue

                examples.append(
                    CombatActionScoreTrainingExample(
                        source_path=str(path),
                        instance_id=instance_id,
                        decision_index=decision_index,
                        decision_point_id=decision_point_id,
                        action_id=action_id,
                        action=dict(action),
                        features=combat_action_score_features(dto, action),
                        estimated_q=node_score,
                        target_source=target_source,
                        sample_weight=weights[target_source],
                        censored=censored,
                        censor_reason=censor_reason,
                        terminal_reached=terminal_reached,
                        dto_version=contract.dto_version,
                    )
                )

    return examples, CombatActionScoreDatasetStats(
        decision_records=decision_records,
        root_actions=root_actions,
        usable_actions=len(examples),
        no_target_actions=no_target_actions,
        unresolved_actions=unresolved_actions,
        censored_actions=censored_actions,
        dto_version=contract.dto_version,
    )


def build_pairwise_action_score_examples(
    examples: Sequence[CombatActionScoreTrainingExample],
    *,
    tie_tolerance: float = 1e-9,
) -> list[PairwiseActionScoreExample]:
    """Convert per-action Oracle node scores into symmetric within-decision rankings."""

    if not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be a finite non-negative number")
    by_decision: dict[tuple[str, str, int], list[CombatActionScoreTrainingExample]] = defaultdict(list)
    for example in examples:
        by_decision[example.decision_key].append(example)

    pairs: list[PairwiseActionScoreExample] = []
    for decision_key, group in by_decision.items():
        for left, right in combinations(group, 2):
            difference = left.node_score - right.node_score
            if abs(difference) <= tie_tolerance:
                continue
            winner, loser = (left, right) if difference > 0.0 else (right, left)
            delta = tuple(
                win - lose
                for win, lose in zip(winner.features, loser.features, strict=True)
            )
            weight = min(winner.sample_weight, loser.sample_weight)
            pairs.append(
                PairwiseActionScoreExample(
                    decision_key=decision_key,
                    winner_action_id=winner.action_id,
                    loser_action_id=loser.action_id,
                    feature_delta=delta,
                    label=1,
                    sample_weight=weight,
                )
            )
            pairs.append(
                PairwiseActionScoreExample(
                    decision_key=decision_key,
                    winner_action_id=winner.action_id,
                    loser_action_id=loser.action_id,
                    feature_delta=tuple(-value for value in delta),
                    label=0,
                    sample_weight=weight,
                )
            )
    return pairs


def _all_rng_outcomes_resolved(value: Any, *, source: str) -> bool:
    outcomes = _sequence(value, f"{source}.rng_outcomes")
    if not outcomes:
        raise ValueError(f"{source}: labeled action must contain at least one rng_outcome")
    resolved = True
    for index, outcome in enumerate(outcomes):
        field = f"{source}.rng_outcomes[{index}]"
        if not isinstance(outcome, Mapping):
            raise ValueError(f"{field}: outcome must be an object")
        target_source = _string(outcome.get("target_source"), f"{field}.target_source")
        raw_node_score = outcome.get("value")
        if target_source == "no_target" or raw_node_score is None:
            resolved = False
            continue
        if target_source not in {"terminal", "value_bootstrap"}:
            raise ValueError(f"{field}: unsupported target_source {target_source!r}")
        _finite_float(raw_node_score, f"{field}.value")
    return resolved


def _records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"{path}:{line_number}: Oracle JSONL record must be an object")
            yield line_number, record


def _positive_weight(value: Any, field: str) -> float:
    number = _finite_float(value, field)
    if number <= 0.0:
        raise ValueError(f"{field} must be positive")
    return number


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a sequence")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field)


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a bool")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


__all__ = [
    "CombatActionScoreDatasetStats",
    "CombatActionScoreTrainingExample",
    "PairwiseActionScoreExample",
    "build_pairwise_action_score_examples",
    "load_combat_action_score_examples",
]
