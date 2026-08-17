"""Build supervised Combat ValueModel examples from Oracle v7/v2 actual episodes."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sts2_training.api.contract import SCHEMA_VERSION
from sts2_training.decision.oracle_log import (
    ORACLE_EPISODE_RESULT_SCHEMA_VERSION,
    ORACLE_RECORD_SCHEMA_VERSION,
    ORACLE_VALUE_MASK_VERSION,
    oracle_value_dto_contract,
)
from sts2_training.decision.oracle_teacher_provenance import inspect_oracle_teacher_provenance
from sts2_training.decision.value_features import (
    VALUE_FEATURE_NAMES,
    ValueFeatureContext,
    combat_value_features,
)


@dataclass(frozen=True)
class OracleValueDtoContract:
    wire_schema_version: str
    mask_version: str
    dto_version: str

    def to_json(self) -> dict[str, str]:
        return {
            "wire_schema_version": self.wire_schema_version,
            "mask_version": self.mask_version,
            "dto_version": self.dto_version,
        }


@dataclass(frozen=True)
class CombatValueTrainingExample:
    source_path: str
    instance_id: str
    decision_index: int
    decision_point_id: str
    masked_emulator_dto: Mapping[str, Any]
    features: tuple[float, ...]
    target_value: float
    target_source: str
    sample_weight: float
    terminal: bool
    dto_version: str
    return_horizon: int | None = None


@dataclass(frozen=True)
class CombatValueDatasetStats:
    decision_records: int
    usable_examples: int
    skipped_records: int
    terminal_examples: int
    bootstrap_examples: int
    dto_version: str | None = None
    terminal_return_examples: int = 0
    truncated_td_examples: int = 0
    incomplete_episodes: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "decision_records": self.decision_records,
            "usable_examples": self.usable_examples,
            "skipped_records": self.skipped_records,
            "terminal_examples": self.terminal_examples,
            "bootstrap_examples": self.bootstrap_examples,
            "dto_version": self.dto_version,
            "terminal_return_examples": self.terminal_return_examples,
            "truncated_td_examples": self.truncated_td_examples,
            "incomplete_episodes": self.incomplete_episodes,
        }


@dataclass(frozen=True)
class _DecisionRow:
    source_path: str
    instance_id: str
    decision_index: int
    decision_point_id: str
    dto: Mapping[str, Any]
    transition: Mapping[str, Any]


@dataclass(frozen=True)
class _EpisodeResult:
    source_path: str
    instance_id: str
    decisions_collected: int
    completed: bool
    termination_reason: str
    combat_result: str | None
    final_dto: Mapping[str, Any]


def inspect_oracle_value_dto_contract(
    paths: Iterable[str | Path],
) -> OracleValueDtoContract:
    normalized_paths = tuple(Path(path) for path in paths)
    if not normalized_paths:
        raise ValueError("Oracle Value data requires at least one JSONL path")

    observed: dict[tuple[str, str, str], list[str]] = {}
    for path in normalized_paths:
        for line_number, record in _records(path):
            record_type = record.get("record_type")
            if record_type not in {"combat_oracle_decision", "combat_oracle_episode_result"}:
                continue
            if record_type == "combat_oracle_decision":
                expected_schema = ORACLE_RECORD_SCHEMA_VERSION
                dto_field = "masked_emulator_dto"
                label = "decision"
            else:
                expected_schema = ORACLE_EPISODE_RESULT_SCHEMA_VERSION
                dto_field = "final_masked_emulator_dto"
                label = "episode result"
            actual_schema = record.get("record_schema_version")
            if actual_schema != expected_schema:
                raise ValueError(
                    f"{path}:{line_number}: expected Oracle {label} schema "
                    f"v{expected_schema}, got {actual_schema!r}"
                )
            dto = _mapping(
                record.get(dto_field),
                f"{path}:{line_number}.{dto_field}",
            )
            wire = oracle_value_dto_contract(
                dto,
                context=f"{path}:{line_number}.{dto_field}",
            )
            declared = record.get("dto_contract")
            if not isinstance(declared, Mapping):
                raise ValueError(f"{path}:{line_number}.dto_contract must be an object")
            declared_triplet = {
                "wire_schema_version": declared.get("wire_schema_version"),
                "mask_version": declared.get("mask_version"),
                "dto_version": declared.get("dto_version"),
            }
            if declared_triplet != wire:
                raise ValueError(
                    f"{path}:{line_number}: dto_contract {declared_triplet!r} does not match "
                    f"embedded DTO contract {wire!r}"
                )
            key = (
                wire["wire_schema_version"],
                wire["mask_version"],
                wire["dto_version"],
            )
            observed.setdefault(key, []).append(f"{path}:{line_number}")

    if not observed:
        raise ValueError("no Oracle Combat decision/episode records found")
    if len(observed) != 1:
        details = "; ".join(
            f"wire={wire!r} mask={mask!r} dto={dto!r} at {locations[:3]!r}"
            for (wire, mask, dto), locations in sorted(observed.items())
        )
        raise ValueError(f"mixed Oracle DTO contracts are not supported: {details}")
    (wire_schema_version, mask_version, dto_version), _locations = next(iter(observed.items()))
    return OracleValueDtoContract(
        wire_schema_version=wire_schema_version,
        mask_version=mask_version,
        dto_version=dto_version,
    )


def load_combat_value_examples(
    paths: Iterable[str | Path],
    *,
    terminal_win_value: float = 100.0,
    terminal_loss_value: float = -100.0,
    gamma: float = 1.0,
    terminal_weight: float = 1.0,
    bootstrap_weight: float = 0.5,
    allow_mixed_teachers: bool = False,
) -> tuple[list[CombatValueTrainingExample], CombatValueDatasetStats]:
    normalized_paths = tuple(Path(path) for path in paths)
    if not normalized_paths:
        raise ValueError("Combat Value training requires at least one Oracle JSONL path")
    gamma_value = _finite_float(gamma, "gamma")
    if gamma_value < 0.0 or gamma_value > 1.0:
        raise ValueError("gamma must be in [0, 1]")
    terminal_win = _finite_float(terminal_win_value, "terminal_win_value")
    terminal_loss = _finite_float(terminal_loss_value, "terminal_loss_value")
    terminal_weight_value = _positive_weight(terminal_weight, "terminal_weight")
    bootstrap_weight_value = _positive_weight(bootstrap_weight, "bootstrap_weight")

    contract = inspect_oracle_value_dto_contract(normalized_paths)
    if contract.wire_schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Combat Value training requires wire schema {SCHEMA_VERSION!r}; "
            f"got {contract.wire_schema_version!r}"
        )
    if contract.mask_version != ORACLE_VALUE_MASK_VERSION:
        raise ValueError(
            f"Combat Value training requires mask {ORACLE_VALUE_MASK_VERSION!r}; "
            f"got {contract.mask_version!r}"
        )
    inspect_oracle_teacher_provenance(
        normalized_paths,
        allow_mixed_teachers=allow_mixed_teachers,
    )

    rows, results, decision_records = _load_actual_episode_rows(normalized_paths)
    examples: list[CombatValueTrainingExample] = []
    skipped_records = 0
    terminal_examples = 0
    bootstrap_examples = 0
    terminal_return_examples = 0
    truncated_td_examples = 0
    incomplete_episodes = 0

    by_instance: dict[str, list[_DecisionRow]] = {}
    for row in rows:
        by_instance.setdefault(row.instance_id, []).append(row)
    instance_ids = set(by_instance) | set(results)
    for instance_id in sorted(instance_ids):
        decisions = sorted(by_instance.get(instance_id, []), key=lambda row: row.decision_index)
        episode = results.get(instance_id)
        if episode is None:
            incomplete_episodes += 1
            skipped_records += len(decisions)
            continue
        if episode.decisions_collected != len(decisions):
            raise ValueError(
                f"{episode.source_path}: episode {instance_id!r} says decisions_collected="
                f"{episode.decisions_collected} but loaded {len(decisions)} decision records"
            )
        for expected_index, row in enumerate(decisions):
            if row.decision_index != expected_index:
                raise ValueError(
                    f"{row.source_path}: episode {instance_id!r} has non-contiguous decision_index "
                    f"{row.decision_index}, expected {expected_index}"
                )

        if episode.completed and episode.combat_result in {"victory", "defeat"}:
            final_return = terminal_win if episode.combat_result == "victory" else terminal_loss
            horizon = len(decisions)
            for position, row in enumerate(decisions):
                remaining = horizon - position
                target = (gamma_value ** remaining) * final_return
                context = ValueFeatureContext(
                    remaining_depth=remaining,
                    max_depth=max(horizon, 1),
                    remaining_time_ms=None,
                    time_budget_ms=None,
                )
                features = combat_value_features(row.dto, context=context)
                examples.append(
                    CombatValueTrainingExample(
                        source_path=row.source_path,
                        instance_id=instance_id,
                        decision_index=row.decision_index,
                        decision_point_id=row.decision_point_id,
                        masked_emulator_dto=dict(row.dto),
                        features=features,
                        target_value=target,
                        target_source="terminal_return",
                        sample_weight=terminal_weight_value,
                        terminal=remaining == 1,
                        dto_version=contract.dto_version,
                        return_horizon=remaining,
                    )
                )
                terminal_examples += 1
                terminal_return_examples += 1
            continue

        if episode.completed:
            skipped_records += len(decisions)
            continue

        bootstrap = _bootstrap_value(episode.final_dto)
        if bootstrap is None:
            skipped_records += len(decisions)
            continue
        horizon = len(decisions)
        for position, row in enumerate(decisions):
            remaining = horizon - position
            target = (gamma_value ** remaining) * bootstrap
            context = ValueFeatureContext(
                remaining_depth=remaining,
                max_depth=max(horizon, 1),
                remaining_time_ms=None,
                time_budget_ms=None,
            )
            features = combat_value_features(row.dto, context=context)
            examples.append(
                CombatValueTrainingExample(
                    source_path=row.source_path,
                    instance_id=instance_id,
                    decision_index=row.decision_index,
                    decision_point_id=row.decision_point_id,
                    masked_emulator_dto=dict(row.dto),
                    features=features,
                    target_value=target,
                    target_source="truncated_td",
                    sample_weight=bootstrap_weight_value,
                    terminal=False,
                    dto_version=contract.dto_version,
                    return_horizon=remaining,
                )
            )
            bootstrap_examples += 1
            truncated_td_examples += 1

    return examples, CombatValueDatasetStats(
        decision_records=decision_records,
        usable_examples=len(examples),
        skipped_records=skipped_records,
        terminal_examples=terminal_examples,
        bootstrap_examples=bootstrap_examples,
        dto_version=contract.dto_version,
        terminal_return_examples=terminal_return_examples,
        truncated_td_examples=truncated_td_examples,
        incomplete_episodes=incomplete_episodes,
    )


def _load_actual_episode_rows(
    paths: Sequence[Path],
) -> tuple[list[_DecisionRow], dict[str, _EpisodeResult], int]:
    rows: list[_DecisionRow] = []
    results: dict[str, _EpisodeResult] = {}
    decision_records = 0
    for path in paths:
        for line_number, record in _records(path):
            record_type = record.get("record_type")
            source = f"{path}:{line_number}"
            if record_type == "combat_oracle_decision":
                decision_records += 1
                instance_id = _string(record.get("instance_id"), f"{source}.instance_id")
                decision_index = _nonnegative_int(
                    record.get("decision_index"),
                    f"{source}.decision_index",
                )
                decision_point_id = _string(
                    record.get("decision_point_id"),
                    f"{source}.decision_point_id",
                )
                dto = _mapping(
                    record.get("masked_emulator_dto"),
                    f"{source}.masked_emulator_dto",
                )
                transition = _mapping(
                    record.get("runtime_transition"),
                    f"{source}.runtime_transition",
                )
                rows.append(
                    _DecisionRow(
                        source_path=source,
                        instance_id=instance_id,
                        decision_index=decision_index,
                        decision_point_id=decision_point_id,
                        dto=dict(dto),
                        transition=dict(transition),
                    )
                )
            elif record_type == "combat_oracle_episode_result":
                instance_id = _string(record.get("instance_id"), f"{source}.instance_id")
                if instance_id in results:
                    raise ValueError(f"duplicate episode result for instance_id={instance_id!r}")
                final_dto = _mapping(
                    record.get("final_masked_emulator_dto"),
                    f"{source}.final_masked_emulator_dto",
                )
                results[instance_id] = _EpisodeResult(
                    source_path=source,
                    instance_id=instance_id,
                    decisions_collected=_nonnegative_int(
                        record.get("decisions_collected"),
                        f"{source}.decisions_collected",
                    ),
                    completed=_bool(record.get("completed"), f"{source}.completed"),
                    termination_reason=_string(
                        record.get("termination_reason"),
                        f"{source}.termination_reason",
                    ),
                    combat_result=_optional_string(
                        record.get("combat_result"),
                        f"{source}.combat_result",
                    ),
                    final_dto=dict(final_dto),
                )
    return rows, results, decision_records


def _bootstrap_value(dto: Mapping[str, Any]) -> float | None:
    candidates = [
        dto.get("value"),
        dto.get("state_value"),
        dto.get("value_estimate"),
    ]
    for candidate in candidates:
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
            continue
        value = float(candidate)
        if math.isfinite(value):
            return value
    return None


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


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
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
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _positive_weight(value: float, field: str) -> float:
    result = _finite_float(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result
