"""Load supervised and actual-trajectory Combat ValueModel data from Oracle v6 JSONL."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sts2_training.decision.oracle_log import (
    ORACLE_EPISODE_RESULT_SCHEMA_VERSION,
    ORACLE_RECORD_SCHEMA_VERSION,
    ORACLE_VALUE_MASK_VERSION,
    oracle_value_dto_contract,
    require_oracle_value_dto_version,
)
from sts2_training.decision.oracle_teacher_provenance import inspect_oracle_teacher_provenance
from sts2_training.decision.value_features import combat_card_summary, combat_value_features


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
    root_decision_point_id: str
    decision_point_id: str | None
    action_id: str
    action: Mapping[str, Any]
    rng_id: int
    root_state_node_id: str
    features: tuple[float, ...]
    target_value: float
    target_source: str
    sample_weight: float
    terminal_reached: bool
    deepest_combat_depth: int
    censored: bool
    censor_reason: str | None
    best_node_id: str | None
    masked_emulator_dto: Mapping[str, Any]
    dto_version: str


@dataclass(frozen=True)
class CombatValueDatasetStats:
    decision_records: int
    root_samples: int
    usable_samples: int
    no_target_samples: int
    upgraded_card_instances: int = 0
    enchanted_card_instances: int = 0
    examples_with_upgrade: int = 0
    examples_with_enchantment: int = 0
    dto_version: str | None = None

    @property
    def label_coverage(self) -> float | None:
        return None if self.root_samples == 0 else self.usable_samples / self.root_samples

    def to_json(self) -> dict[str, Any]:
        return {
            "decision_records": self.decision_records,
            "root_samples": self.root_samples,
            "usable_samples": self.usable_samples,
            "no_target_samples": self.no_target_samples,
            "label_coverage": self.label_coverage,
            "upgraded_card_instances": self.upgraded_card_instances,
            "enchanted_card_instances": self.enchanted_card_instances,
            "examples_with_upgrade": self.examples_with_upgrade,
            "examples_with_enchantment": self.examples_with_enchantment,
            "dto_version": self.dto_version,
        }


@dataclass(frozen=True)
class CombatValueRLTimestep:
    """One actual runtime transition, distinct from counterfactual Oracle root samples."""

    source_path: str
    instance_id: str
    server_epoch: str | None
    decision_index: int
    decision_point_id: str
    decision_response_metadata: Mapping[str, Any]
    masked_emulator_dto: Mapping[str, Any]
    features: tuple[float, ...]
    chosen_action_id: str
    chosen_action: Mapping[str, Any]
    next_decision_point_id: str
    commit_response_metadata: Mapping[str, Any]
    next_masked_emulator_dto: Mapping[str, Any]
    combat_result_after_step: str | None
    dto_version: str


@dataclass(frozen=True)
class CombatValueRLEpisode:
    """Actual committed Combat trajectory plus its terminal/truncated collection result."""

    source_paths: tuple[str, ...]
    instance_id: str
    server_epoch: str | None
    steps: tuple[CombatValueRLTimestep, ...]
    completed: bool
    termination_reason: str
    combat_result: str | None
    final_decision_metadata: Mapping[str, Any]
    final_masked_emulator_dto: Mapping[str, Any]
    dto_version: str

    @property
    def usable_for_terminal_return(self) -> bool:
        return self.completed and self.combat_result in {"victory", "defeat"}


def inspect_oracle_value_dto_contract(
    paths: Iterable[str | Path],
) -> OracleValueDtoContract:
    """Require one exact wire/mask/Emulator DTO generation across all Oracle records."""

    normalized_paths = tuple(Path(path) for path in paths)
    if not normalized_paths:
        raise ValueError("Combat Value data requires at least one Oracle JSONL path")
    contracts: set[tuple[str, str, str]] = set()
    seen = 0
    for path in normalized_paths:
        for line_number, record in _records(path):
            record_type = record.get("record_type")
            if record_type == "combat_oracle_decision":
                if record.get("record_schema_version") != ORACLE_RECORD_SCHEMA_VERSION:
                    raise ValueError(
                        f"{path}:{line_number}: expected Oracle decision schema "
                        f"v{ORACLE_RECORD_SCHEMA_VERSION}, got "
                        f"{record.get('record_schema_version')!r}"
                    )
                dto = _mapping(
                    record.get("masked_emulator_dto"),
                    f"{path}:{line_number}.masked_emulator_dto",
                )
            elif record_type == "combat_oracle_episode_result":
                if record.get("record_schema_version") != ORACLE_EPISODE_RESULT_SCHEMA_VERSION:
                    raise ValueError(
                        f"{path}:{line_number}: expected Oracle episode schema "
                        f"v{ORACLE_EPISODE_RESULT_SCHEMA_VERSION}, got "
                        f"{record.get('record_schema_version')!r}"
                    )
                dto = _mapping(
                    record.get("final_masked_emulator_dto"),
                    f"{path}:{line_number}.final_masked_emulator_dto",
                )
            else:
                continue
            seen += 1
            actual = oracle_value_dto_contract(
                dto,
                context=f"{path}:{line_number} {record_type}",
            )
            declared = _mapping(
                record.get("dto_contract"),
                f"{path}:{line_number}.dto_contract",
            )
            declared_tuple = (
                _string(declared.get("wire_schema_version"), f"{path}:{line_number}.dto_contract.wire_schema_version"),
                _string(declared.get("mask_version"), f"{path}:{line_number}.dto_contract.mask_version"),
                _string(declared.get("dto_version"), f"{path}:{line_number}.dto_contract.dto_version"),
            )
            actual_tuple = (
                actual["wire_schema_version"],
                actual["mask_version"],
                actual["dto_version"],
            )
            if declared_tuple != actual_tuple:
                raise ValueError(
                    f"{path}:{line_number}: dto_contract does not match embedded masked DTO"
                )
            contracts.add(actual_tuple)
    if seen == 0:
        raise ValueError("no Oracle decision/episode records found")
    if len(contracts) != 1:
        raise ValueError(
            "Combat Value dataset mixes incompatible wire/mask/dto generations: "
            f"{sorted(contracts)!r}"
        )
    wire_schema_version, mask_version, dto_version = next(iter(contracts))
    return OracleValueDtoContract(
        wire_schema_version=wire_schema_version,
        mask_version=mask_version,
        dto_version=dto_version,
    )


def load_combat_value_examples(
    paths: Iterable[str | Path],
    *,
    terminal_weight: float = 1.0,
    bootstrap_weight: float = 0.5,
    allow_mixed_teachers: bool = False,
) -> tuple[list[CombatValueTrainingExample], CombatValueDatasetStats]:
    """Load Oracle root-action targets for supervised Value distillation.

    Counterfactual root samples remain separate from actual committed runtime transitions.
    ``no_target`` is retained in coverage counts but never converted to a numeric label.
    """

    if terminal_weight <= 0.0 or bootstrap_weight <= 0.0:
        raise ValueError("target source weights must be positive")
    normalized_paths = tuple(Path(path) for path in paths)
    if not normalized_paths:
        raise ValueError("Combat Value training requires at least one Oracle JSONL path")
    contract = inspect_oracle_value_dto_contract(normalized_paths)
    inspect_oracle_teacher_provenance(
        normalized_paths,
        allow_mixed_teachers=allow_mixed_teachers,
    )

    examples: list[CombatValueTrainingExample] = []
    decision_records = 0
    root_samples = 0
    no_target_samples = 0
    upgraded_card_instances = 0
    enchanted_card_instances = 0
    examples_with_upgrade = 0
    examples_with_enchantment = 0
    for path in normalized_paths:
        for line_number, record in _records(path):
            if record.get("record_type") != "combat_oracle_decision":
                continue
            decision_records += 1
            source_prefix = f"{path}:{line_number}"
            if record.get("record_schema_version") != ORACLE_RECORD_SCHEMA_VERSION:
                raise ValueError(
                    f"{source_prefix}: expected Oracle record schema "
                    f"v{ORACLE_RECORD_SCHEMA_VERSION}, got "
                    f"{record.get('record_schema_version')!r}"
                )
            instance_id = _string(record.get("instance_id"), f"{source_prefix}.instance_id")
            decision_index = _nonnegative_int(
                record.get("decision_index"), f"{source_prefix}.decision_index"
            )
            root_decision_point_id = _string(
                record.get("decision_point_id"), f"{source_prefix}.decision_point_id"
            )
            decision_dto = _mapping(
                record.get("masked_emulator_dto"), f"{source_prefix}.masked_emulator_dto"
            )
            require_oracle_value_dto_version(
                decision_dto,
                expected=contract.dto_version,
                context=f"{source_prefix}.masked_emulator_dto",
            )
            _validate_runtime_transition(
                record,
                source_prefix=source_prefix,
                dto_version=contract.dto_version,
            )
            raw_samples = record.get("root_value_samples")
            if not isinstance(raw_samples, Sequence) or isinstance(
                raw_samples, (str, bytes, bytearray)
            ):
                raise ValueError(f"{source_prefix}: root_value_samples must be a sequence")
            seen: set[tuple[str, int]] = set()
            for sample_index, raw in enumerate(raw_samples):
                source = f"{source_prefix}.root_value_samples[{sample_index}]"
                if not isinstance(raw, Mapping):
                    raise ValueError(f"{source}: sample must be an object")
                root_samples += 1
                action_id = _string(raw.get("action_id"), f"{source}.action_id")
                action = _mapping(raw.get("action"), f"{source}.action")
                if action.get("action_id") not in {None, action_id}:
                    raise ValueError(f"{source}.action.action_id does not match action_id")
                rng_id = _int(raw.get("rng_id"), f"{source}.rng_id")
                key = (action_id, rng_id)
                if key in seen:
                    raise ValueError(f"{source}: duplicate action_id/rng_id sample {key!r}")
                seen.add(key)
                root_state_node_id = _string(
                    raw.get("root_state_node_id"), f"{source}.root_state_node_id"
                )
                dto = _mapping(raw.get("masked_emulator_dto"), f"{source}.masked_emulator_dto")
                require_oracle_value_dto_version(
                    dto,
                    expected=contract.dto_version,
                    context=f"{source}.masked_emulator_dto",
                )
                sample_decision_point_id = _post_state_decision_point_id(
                    raw.get("decision_point_id"),
                    terminal=_is_terminal_post_state(dto),
                    field=f"{source}.decision_point_id",
                )
                deepest_combat_depth = _nonnegative_int(
                    raw.get("deepest_combat_depth"), f"{source}.deepest_combat_depth"
                )
                card_summary = combat_card_summary(dto)
                upgraded_card_instances += card_summary.upgraded_card_count
                enchanted_card_instances += card_summary.enchanted_card_count
                if card_summary.upgraded_card_count:
                    examples_with_upgrade += 1
                if card_summary.enchanted_card_count:
                    examples_with_enchantment += 1

                target_source = _string(raw.get("target_source"), f"{source}.target_source")
                target_value = raw.get("target_value")
                if target_source == "no_target":
                    if target_value is not None:
                        raise ValueError(f"{source}: no_target sample must have null target_value")
                    no_target_samples += 1
                    continue
                if target_source not in {"terminal", "value_bootstrap"}:
                    raise ValueError(f"{source}: unsupported target_source {target_source!r}")
                value = _finite_float(target_value, f"{source}.target_value")
                sample_weight = terminal_weight if target_source == "terminal" else bootstrap_weight
                terminal_reached = _bool(raw.get("terminal_reached"), f"{source}.terminal_reached")
                censored = _bool(raw.get("censored"), f"{source}.censored")
                censor_reason = _optional_string(raw.get("censor_reason"), f"{source}.censor_reason")
                best_node_id = _optional_string(raw.get("best_node_id"), f"{source}.best_node_id")
                examples.append(
                    CombatValueTrainingExample(
                        source_path=str(path),
                        instance_id=instance_id,
                        decision_index=decision_index,
                        root_decision_point_id=root_decision_point_id,
                        decision_point_id=sample_decision_point_id,
                        action_id=action_id,
                        action=dict(action),
                        rng_id=rng_id,
                        root_state_node_id=root_state_node_id,
                        features=combat_value_features(dto),
                        target_value=value,
                        target_source=target_source,
                        sample_weight=sample_weight,
                        terminal_reached=terminal_reached,
                        deepest_combat_depth=deepest_combat_depth,
                        censored=censored,
                        censor_reason=censor_reason,
                        best_node_id=best_node_id,
                        masked_emulator_dto=dict(dto),
                        dto_version=contract.dto_version,
                    )
                )

    return examples, CombatValueDatasetStats(
        decision_records=decision_records,
        root_samples=root_samples,
        usable_samples=len(examples),
        no_target_samples=no_target_samples,
        upgraded_card_instances=upgraded_card_instances,
        enchanted_card_instances=enchanted_card_instances,
        examples_with_upgrade=examples_with_upgrade,
        examples_with_enchantment=examples_with_enchantment,
        dto_version=contract.dto_version,
    )


def load_combat_value_rl_episodes(
    paths: Iterable[str | Path],
    *,
    completed_only: bool = False,
) -> list[CombatValueRLEpisode]:
    """Load actual committed trajectories without inventing an RL reward/objective.

    The base loader deliberately returns both completed and deliberately truncated
    episodes.  Consumers that use terminal Monte-Carlo returns can request
    ``completed_only=True`` or filter by ``usable_for_terminal_return``. Counterfactual
    ``root_value_samples`` are never treated as actions that actually occurred.
    """

    normalized_paths = tuple(Path(path) for path in paths)
    contract = inspect_oracle_value_dto_contract(normalized_paths)
    steps_by_key: dict[tuple[str | None, str], list[CombatValueRLTimestep]] = {}
    result_by_key: dict[tuple[str | None, str], tuple[str, Mapping[str, Any]]] = {}
    paths_by_key: dict[tuple[str | None, str], set[str]] = {}

    for path in normalized_paths:
        for line_number, record in _records(path):
            record_type = record.get("record_type")
            source_prefix = f"{path}:{line_number}"
            if record_type == "combat_oracle_decision":
                if record.get("record_schema_version") != ORACLE_RECORD_SCHEMA_VERSION:
                    raise ValueError(f"{source_prefix}: incompatible Oracle decision schema")
                instance_id = _string(record.get("instance_id"), f"{source_prefix}.instance_id")
                metadata = _mapping(
                    record.get("decision_response_metadata"),
                    f"{source_prefix}.decision_response_metadata",
                )
                server_epoch = _optional_string(metadata.get("server_epoch"), f"{source_prefix}.server_epoch")
                key = (server_epoch, instance_id)
                transition = _validate_runtime_transition(
                    record,
                    source_prefix=source_prefix,
                    dto_version=contract.dto_version,
                )
                dto = _mapping(record.get("masked_emulator_dto"), f"{source_prefix}.masked_emulator_dto")
                require_oracle_value_dto_version(
                    dto,
                    expected=contract.dto_version,
                    context=f"{source_prefix}.masked_emulator_dto",
                )
                step = CombatValueRLTimestep(
                    source_path=str(path),
                    instance_id=instance_id,
                    server_epoch=server_epoch,
                    decision_index=_nonnegative_int(
                        record.get("decision_index"), f"{source_prefix}.decision_index"
                    ),
                    decision_point_id=_string(
                        record.get("decision_point_id"), f"{source_prefix}.decision_point_id"
                    ),
                    decision_response_metadata=dict(metadata),
                    masked_emulator_dto=dict(dto),
                    features=combat_value_features(dto),
                    chosen_action_id=transition["chosen_action_id"],
                    chosen_action=dict(transition["chosen_action"]),
                    next_decision_point_id=transition["next_decision_point_id"],
                    commit_response_metadata=dict(transition["commit_response_metadata"]),
                    next_masked_emulator_dto=dict(transition["next_masked_emulator_dto"]),
                    combat_result_after_step=transition.get("combat_result"),
                    dto_version=contract.dto_version,
                )
                steps_by_key.setdefault(key, []).append(step)
                paths_by_key.setdefault(key, set()).add(str(path))
            elif record_type == "combat_oracle_episode_result":
                if record.get("record_schema_version") != ORACLE_EPISODE_RESULT_SCHEMA_VERSION:
                    raise ValueError(f"{source_prefix}: incompatible Oracle episode schema")
                instance_id = _string(record.get("instance_id"), f"{source_prefix}.instance_id")
                final_metadata = _mapping(
                    record.get("final_decision_metadata"), f"{source_prefix}.final_decision_metadata"
                )
                server_epoch = _optional_string(
                    final_metadata.get("server_epoch"), f"{source_prefix}.server_epoch"
                )
                key = (server_epoch, instance_id)
                if key in result_by_key:
                    raise ValueError(f"{source_prefix}: duplicate episode result for {key!r}")
                result_by_key[key] = (str(path), record)
                paths_by_key.setdefault(key, set()).add(str(path))

    episodes: list[CombatValueRLEpisode] = []
    for key in sorted(set(steps_by_key) | set(result_by_key), key=lambda item: (str(item[0]), item[1])):
        if key not in result_by_key:
            raise ValueError(f"actual trajectory {key!r} is missing combat_oracle_episode_result")
        result_path, record = result_by_key[key]
        instance_id = key[1]
        steps = sorted(steps_by_key.get(key, []), key=lambda step: step.decision_index)
        for expected_index, step in enumerate(steps):
            if step.decision_index != expected_index:
                raise ValueError(
                    f"actual trajectory {key!r} has non-contiguous decision_index values"
                )
        decisions_collected = _nonnegative_int(
            record.get("decisions_collected"), f"{result_path}.decisions_collected"
        )
        if decisions_collected != len(steps):
            raise ValueError(
                f"actual trajectory {key!r} decisions_collected={decisions_collected} "
                f"but loaded {len(steps)} decision records"
            )
        final_dto = _mapping(
            record.get("final_masked_emulator_dto"), f"{result_path}.final_masked_emulator_dto"
        )
        require_oracle_value_dto_version(
            final_dto,
            expected=contract.dto_version,
            context=f"{result_path}.final_masked_emulator_dto",
        )
        if steps and dict(steps[-1].next_masked_emulator_dto) != dict(final_dto):
            raise ValueError(f"actual trajectory {key!r} final DTO does not match last transition")
        completed = _bool(record.get("completed"), f"{result_path}.completed")
        termination_reason = _string(
            record.get("termination_reason"), f"{result_path}.termination_reason"
        )
        combat_result = record.get("combat_result")
        if combat_result is not None and combat_result not in {"victory", "defeat"}:
            raise ValueError(f"{result_path}.combat_result must be victory/defeat/null")
        episode = CombatValueRLEpisode(
            source_paths=tuple(sorted(paths_by_key.get(key, {result_path}))),
            instance_id=instance_id,
            server_epoch=key[0],
            steps=tuple(steps),
            completed=completed,
            termination_reason=termination_reason,
            combat_result=combat_result,
            final_decision_metadata=dict(
                _mapping(record.get("final_decision_metadata"), f"{result_path}.final_decision_metadata")
            ),
            final_masked_emulator_dto=dict(final_dto),
            dto_version=contract.dto_version,
        )
        if not completed_only or episode.usable_for_terminal_return:
            episodes.append(episode)
    return episodes


def _validate_runtime_transition(
    record: Mapping[str, Any],
    *,
    source_prefix: str,
    dto_version: str,
) -> dict[str, Any]:
    transition = _mapping(record.get("runtime_transition"), f"{source_prefix}.runtime_transition")
    chosen_action_id = _string(
        transition.get("chosen_action_id"), f"{source_prefix}.runtime_transition.chosen_action_id"
    )
    chosen_action = _mapping(
        transition.get("chosen_action"), f"{source_prefix}.runtime_transition.chosen_action"
    )
    if chosen_action.get("action_id") not in {None, chosen_action_id}:
        raise ValueError(f"{source_prefix}.runtime_transition chosen action id mismatch")
    next_decision_point_id = _string(
        transition.get("next_decision_point_id"),
        f"{source_prefix}.runtime_transition.next_decision_point_id",
    )
    commit_metadata = _mapping(
        transition.get("commit_response_metadata"),
        f"{source_prefix}.runtime_transition.commit_response_metadata",
    )
    next_dto = _mapping(
        transition.get("next_masked_emulator_dto"),
        f"{source_prefix}.runtime_transition.next_masked_emulator_dto",
    )
    require_oracle_value_dto_version(
        next_dto,
        expected=dto_version,
        context=f"{source_prefix}.runtime_transition.next_masked_emulator_dto",
    )
    combat_result = transition.get("combat_result")
    if combat_result is not None and combat_result not in {"victory", "defeat"}:
        raise ValueError(f"{source_prefix}.runtime_transition.combat_result is invalid")
    return {
        "chosen_action_id": chosen_action_id,
        "chosen_action": chosen_action,
        "next_decision_point_id": next_decision_point_id,
        "commit_response_metadata": commit_metadata,
        "next_masked_emulator_dto": next_dto,
        "combat_result": combat_result,
    }


def _post_state_decision_point_id(
    value: Any,
    *,
    terminal: bool,
    field: str,
) -> str | None:
    if value is None or value == "":
        if terminal:
            return None
        raise ValueError(f"{field} must be a non-empty string for a non-terminal post-state")
    return _string(value, field)


def _is_terminal_post_state(dto: Mapping[str, Any]) -> bool:
    if dto.get("terminal") is True or dto.get("run_terminal") is True:
        return True
    boundary = dto.get("boundary")
    if isinstance(boundary, str) and boundary in {"terminal", "run_terminal"}:
        return True
    transition = dto.get("transition")
    return isinstance(transition, Mapping) and transition.get("kind") == "combat_completed"


def _records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
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


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    result = _int(value, field)
    if result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a bool")
    return value


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


__all__ = [
    "CombatValueDatasetStats",
    "CombatValueRLEpisode",
    "CombatValueRLTimestep",
    "CombatValueTrainingExample",
    "OracleValueDtoContract",
    "inspect_oracle_value_dto_contract",
    "load_combat_value_examples",
    "load_combat_value_rl_episodes",
]
