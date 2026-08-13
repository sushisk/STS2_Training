"""Stable JSONL records for budgeted Combat oracle collection."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from sts2_training.decision.oracle_search import OracleCollectionResult

# v5 adds bounded root-action ValueModel training samples. The raw DTO persisted for a
# sample is exactly the resolved masked_emulator_dto consumed by ValueModel; deeper Beam
# branch DTOs are intentionally not serialized.
ORACLE_RECORD_SCHEMA_VERSION = 5
ORACLE_EPISODE_RESULT_SCHEMA_VERSION = 1
# Value training relies on the full public card-instance identity added by STS2_RL mask
# v1.2: pile multisets retain upgradeLevel, tinker-time state, and enchantment.
ORACLE_VALUE_MASK_VERSION = "1.2"


def require_oracle_value_mask_version(
    dto: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Fail closed when a DTO cannot preserve the card identity needed by ValueModel."""

    version = dto.get("mask_version")
    if version != ORACLE_VALUE_MASK_VERSION:
        raise ValueError(
            f"{context} requires masked DTO mask_version={ORACLE_VALUE_MASK_VERSION!r}; "
            f"got {version!r}"
        )


def _require_root_value_sample_mask_versions(samples: Any) -> None:
    if samples is None:
        return
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes, bytearray)):
        raise ValueError("root_value_samples must be a sequence")
    for index, sample in enumerate(samples):
        if isinstance(sample, Mapping):
            dto = sample.get("masked_emulator_dto")
        else:
            dto = getattr(sample, "masked_emulator_dto", None)
        if not isinstance(dto, Mapping):
            raise ValueError(
                f"Oracle root_value_samples[{index}] must contain masked_emulator_dto"
            )
        require_oracle_value_mask_version(
            dto,
            context=f"Oracle root_value_samples[{index}]",
        )


def oracle_collection_record(
    root_decision: Mapping[str, Any],
    result: OracleCollectionResult,
    *,
    training_commit: str | None = None,
) -> dict[str, Any]:
    """Build one self-contained, re-featurizable record for one root Decision."""

    decision_point_id = root_decision.get("decision_point_id")
    dto = root_decision.get("masked_emulator_dto")
    if not isinstance(decision_point_id, str) or not decision_point_id:
        raise ValueError("root_decision must contain a non-empty decision_point_id")
    if not isinstance(dto, Mapping):
        raise ValueError("root_decision must contain masked_emulator_dto")
    require_oracle_value_mask_version(dto, context="Oracle decision log")
    root_value_samples = getattr(result, "root_value_samples", ())
    _require_root_value_sample_mask_versions(root_value_samples)

    return {
        "record_type": "combat_oracle_decision",
        "record_schema_version": ORACLE_RECORD_SCHEMA_VERSION,
        "decision_point_id": decision_point_id,
        # Keep the raw masked public DTO so future feature extractors can be rebuilt
        # without replaying the emulator or trusting today's feature schema.
        "masked_emulator_dto": _jsonable(dto),
        # Root-only ValueModel data. Older/programmatic OracleCollectionResult instances
        # may omit this attribute, in which case the v5 field is an empty list.
        "root_value_samples": _jsonable(root_value_samples),
        "oracle_search_result": _jsonable(result.search_result),
        "oracle_targets": _jsonable(result.targets),
        "search_trace": [_jsonable(event) for event in result.trace],
        "provenance": {
            "training_commit": training_commit,
            "teacher_policy_class": result.provenance.teacher_policy_class,
            "teacher_inner_policy_class": result.provenance.teacher_inner_policy_class,
            "teacher_coverage_policy_class": (
                result.provenance.teacher_coverage_policy_class
            ),
            "teacher_value_class": result.provenance.teacher_value_class,
            "teacher_policy_metadata": _jsonable(
                result.provenance.teacher_policy_metadata
            ),
            "teacher_inner_policy_metadata": _jsonable(
                result.provenance.teacher_inner_policy_metadata
            ),
            "teacher_value_metadata": _jsonable(
                result.provenance.teacher_value_metadata
            ),
            "pruner_name": result.targets.metadata.pruner_name,
            "pruner_version": result.targets.metadata.pruner_version,
            "rng_sampling": result.targets.metadata.rng_sampling,
        },
    }


def combat_result_from_dto(dto: Mapping[str, Any]) -> str | None:
    """Return a normalized terminal Combat result when the public DTO exposes one."""

    outcome = dto.get("outcome")
    if outcome in {"victory", "run_victory"}:
        return "victory"
    if outcome == "defeat":
        return "defeat"
    transition = dto.get("transition")
    if isinstance(transition, Mapping) and transition.get("kind") == "combat_completed":
        victory = transition.get("victory")
        if victory is True:
            return "victory"
        if victory is False:
            return "defeat"
    return None


def oracle_episode_result_record(
    *,
    instance_id: str,
    decisions_collected: int,
    final_dto: Mapping[str, Any],
    completed: bool,
    termination_reason: str,
    elapsed_s: float,
) -> dict[str, Any]:
    """Build the terminal/truncated episode record appended to the same Oracle JSONL."""

    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("instance_id must be a non-empty string")
    if isinstance(decisions_collected, bool) or not isinstance(decisions_collected, int):
        raise ValueError("decisions_collected must be an integer")
    if decisions_collected < 0:
        raise ValueError("decisions_collected must be non-negative")
    if not isinstance(completed, bool):
        raise TypeError("completed must be a bool")
    if not isinstance(termination_reason, str) or not termination_reason:
        raise ValueError("termination_reason must be a non-empty string")
    if (
        isinstance(elapsed_s, bool)
        or not isinstance(elapsed_s, (int, float))
        or not math.isfinite(float(elapsed_s))
        or elapsed_s < 0
    ):
        raise ValueError("elapsed_s must be a finite non-negative number")
    require_oracle_value_mask_version(final_dto, context="Oracle episode result")

    return {
        "record_type": "combat_oracle_episode_result",
        "record_schema_version": ORACLE_EPISODE_RESULT_SCHEMA_VERSION,
        "instance_id": instance_id,
        "decisions_collected": decisions_collected,
        "completed": completed,
        "termination_reason": termination_reason,
        "combat_result": combat_result_from_dto(final_dto),
        # Persist the final public DTO verbatim so terminal details are not lost even when
        # a future RL schema adds result fields this client does not yet normalize.
        "final_masked_emulator_dto": _jsonable(final_dto),
        "elapsed_s": float(elapsed_s),
    }


class OracleJsonlWriter:
    """Append-only writer; intentionally separate from SelectionAudit JSONL."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(
        self,
        root_decision: Mapping[str, Any],
        result: OracleCollectionResult,
        *,
        training_commit: str | None = None,
    ) -> dict[str, Any]:
        record = oracle_collection_record(
            root_decision,
            result,
            training_commit=training_commit,
        )
        self._append(record)
        return record

    def write_episode_result(
        self,
        *,
        instance_id: str,
        decisions_collected: int,
        final_dto: Mapping[str, Any],
        completed: bool,
        termination_reason: str,
        elapsed_s: float,
    ) -> dict[str, Any]:
        record = oracle_episode_result_record(
            instance_id=instance_id,
            decisions_collected=decisions_collected,
            final_dto=final_dto,
            completed=completed,
            termination_reason=termination_reason,
            elapsed_s=elapsed_s,
        )
        self._append(record)
        return record

    def _append(self, record: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def qualified_class_name(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not JSON serializable by oracle log contract: {type(value)!r}")
