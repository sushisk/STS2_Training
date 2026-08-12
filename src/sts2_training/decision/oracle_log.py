"""Stable JSONL records for budgeted Combat oracle collection."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from sts2_training.decision.oracle_search import OracleCollectionResult

ORACLE_RECORD_SCHEMA_VERSION = 2


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

    return {
        "record_type": "combat_oracle_decision",
        "record_schema_version": ORACLE_RECORD_SCHEMA_VERSION,
        "decision_point_id": decision_point_id,
        # Keep the raw masked public DTO so future feature extractors can be rebuilt
        # without replaying the emulator or trusting today's feature schema.
        "masked_emulator_dto": _jsonable(dto),
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
            "pruner_name": result.targets.metadata.pruner_name,
            "pruner_version": result.targets.metadata.pruner_version,
            "rng_sampling": result.targets.metadata.rng_sampling,
        },
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        return record


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