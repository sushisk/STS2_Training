"""Validate and fingerprint Oracle teacher provenance used for pruner training/eval.

Oracle targets are labels produced by a concrete Policy/Value/search configuration. Mixing
records from different teachers silently changes the supervised objective, so callers must
opt in explicitly when that is intentional. The complete provenance payload is retained in
summaries so trained artifacts and evaluation reports remain auditable.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sts2_training.decision.oracle_log import ORACLE_RECORD_SCHEMA_VERSION

TEACHER_PROVENANCE_SUMMARY_SCHEMA_VERSION = 1
_REQUIRED_PROVENANCE_FIELDS = (
    "teacher_policy_class",
    "teacher_inner_policy_class",
    "teacher_value_class",
    "teacher_policy_metadata",
    "teacher_inner_policy_metadata",
    "teacher_value_metadata",
    "pruner_name",
    "pruner_version",
    "rng_sampling",
)
_METADATA_FIELDS = (
    "teacher_policy_metadata",
    "teacher_inner_policy_metadata",
    "teacher_value_metadata",
)


@dataclass(frozen=True)
class OracleTeacherProvenanceEntry:
    fingerprint_sha256: str
    provenance: dict[str, Any]
    record_count: int
    source_files: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "fingerprint_sha256": self.fingerprint_sha256,
            "record_count": self.record_count,
            "source_files": list(self.source_files),
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class OracleTeacherProvenanceSummary:
    oracle_record_schema_version: int
    teachers: tuple[OracleTeacherProvenanceEntry, ...]
    record_count: int
    source_files: tuple[str, ...]

    @property
    def mixed(self) -> bool:
        return len(self.teachers) > 1

    @property
    def teacher_fingerprints(self) -> tuple[str, ...]:
        return tuple(entry.fingerprint_sha256 for entry in self.teachers)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": TEACHER_PROVENANCE_SUMMARY_SCHEMA_VERSION,
            "oracle_record_schema_version": self.oracle_record_schema_version,
            "mixed": self.mixed,
            "teacher_count": len(self.teachers),
            "record_count": self.record_count,
            "source_files": list(self.source_files),
            "teacher_fingerprints": list(self.teacher_fingerprints),
            "teachers": [entry.to_json() for entry in self.teachers],
        }


def inspect_oracle_teacher_provenance(
    paths: Iterable[str | Path],
    *,
    allow_mixed_teachers: bool = False,
) -> OracleTeacherProvenanceSummary:
    """Inspect Oracle v3 JSONL and reject accidental teacher mixing by default."""

    normalized_paths = tuple(Path(path) for path in paths)
    if not normalized_paths:
        raise ValueError("Oracle teacher provenance inspection requires at least one path")

    by_fingerprint: dict[str, dict[str, Any]] = {}
    total_records = 0
    for path in normalized_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, Mapping):
                    raise ValueError(f"{path}:{line_number}: Oracle JSONL record must be an object")
                if record.get("record_type") != "combat_oracle_decision":
                    continue
                schema_version = record.get("record_schema_version")
                if schema_version != ORACLE_RECORD_SCHEMA_VERSION:
                    raise ValueError(
                        f"{path}:{line_number}: expected Oracle record schema "
                        f"v{ORACLE_RECORD_SCHEMA_VERSION}, got {schema_version!r}"
                    )
                provenance = _validated_provenance(
                    record.get("provenance"), source=f"{path}:{line_number}"
                )
                canonical = _canonical_json(provenance, source=f"{path}:{line_number}.provenance")
                fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                entry = by_fingerprint.setdefault(
                    fingerprint,
                    {
                        "provenance": provenance,
                        "record_count": 0,
                        "source_files": set(),
                    },
                )
                entry["record_count"] += 1
                entry["source_files"].add(str(path))
                total_records += 1

    if total_records == 0:
        raise ValueError("no combat_oracle_decision records found in Oracle JSONL inputs")

    fingerprints = sorted(by_fingerprint)
    if len(fingerprints) > 1 and not allow_mixed_teachers:
        short = ", ".join(value[:12] for value in fingerprints)
        raise ValueError(
            "mixed Oracle teacher provenance detected; pass allow_mixed_teachers=True "
            f"only when intentional (teachers={short})"
        )

    teachers = tuple(
        OracleTeacherProvenanceEntry(
            fingerprint_sha256=fingerprint,
            provenance=dict(by_fingerprint[fingerprint]["provenance"]),
            record_count=int(by_fingerprint[fingerprint]["record_count"]),
            source_files=tuple(sorted(by_fingerprint[fingerprint]["source_files"])),
        )
        for fingerprint in fingerprints
    )
    return OracleTeacherProvenanceSummary(
        oracle_record_schema_version=ORACLE_RECORD_SCHEMA_VERSION,
        teachers=teachers,
        record_count=total_records,
        source_files=tuple(str(path) for path in normalized_paths),
    )


def teacher_provenance_matches(
    artifact_summary: Mapping[str, Any] | None,
    evaluation_summary: OracleTeacherProvenanceSummary,
) -> bool:
    """Return whether artifact-training and evaluation teacher sets are identical."""

    if not isinstance(artifact_summary, Mapping):
        return False
    if artifact_summary.get("oracle_record_schema_version") != ORACLE_RECORD_SCHEMA_VERSION:
        return False
    raw_fingerprints = artifact_summary.get("teacher_fingerprints")
    if not isinstance(raw_fingerprints, Sequence) or isinstance(
        raw_fingerprints, (str, bytes, bytearray)
    ):
        return False
    fingerprints: list[str] = []
    for value in raw_fingerprints:
        if not isinstance(value, str) or not value:
            return False
        fingerprints.append(value)
    return tuple(sorted(fingerprints)) == tuple(sorted(evaluation_summary.teacher_fingerprints))


def require_matching_teacher_provenance(
    artifact_summary: Mapping[str, Any] | None,
    evaluation_summary: OracleTeacherProvenanceSummary,
    *,
    allow_teacher_mismatch: bool = False,
) -> bool:
    """Validate held-out labels came from the same teacher set as the artifact."""

    matches = teacher_provenance_matches(artifact_summary, evaluation_summary)
    if matches or allow_teacher_mismatch:
        return matches
    if artifact_summary is None:
        raise ValueError(
            "learned-pruner artifact does not record Oracle teacher provenance; retrain "
            "with the current trainer or pass allow_teacher_mismatch=True explicitly"
        )
    raise ValueError(
        "evaluation Oracle teacher provenance does not match the teacher set recorded "
        "in the learned-pruner artifact; pass allow_teacher_mismatch=True only when intentional"
    )


def _validated_provenance(value: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{source}: provenance must be an object")
    missing = [field for field in _REQUIRED_PROVENANCE_FIELDS if field not in value]
    if missing:
        raise ValueError(f"{source}: provenance missing required fields: {missing!r}")
    for field in _METADATA_FIELDS:
        if not isinstance(value.get(field), Mapping):
            raise ValueError(f"{source}: provenance.{field} must be an object")
    normalized = _json_value(value, source=f"{source}.provenance")
    if not isinstance(normalized, dict):
        raise AssertionError("validated provenance must normalize to a dict")
    return normalized


def _canonical_json(value: Any, *, source: str) -> str:
    normalized = _json_value(value, source=source)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_value(value: Any, *, source: str) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item, source=f"{source}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item, source=f"{source}[]") for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{source}: provenance contains non-finite float")
        return value
    raise ValueError(f"{source}: provenance contains non-JSON value {type(value)!r}")
