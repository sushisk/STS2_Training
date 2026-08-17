from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sts2_training.decision.oracle_log import ORACLE_RECORD_SCHEMA_VERSION
from sts2_training.decision.oracle_teacher_provenance import (
    inspect_oracle_teacher_provenance,
    require_matching_teacher_provenance,
    teacher_provenance_matches,
)


def _provenance(
    *,
    value_checkpoint: str,
    policy_checkpoint: str = "policy-a",
    pruner_version: str = "1",
) -> dict:
    return {
        "training_commit": "training-commit",
        "teacher_policy_class": "pkg.Policy",
        "teacher_inner_policy_class": "pkg.InnerPolicy",
        "teacher_coverage_policy_class": "pkg.CoveragePolicy",
        "teacher_value_class": "pkg.Value",
        "teacher_policy_metadata": {"checkpoint": policy_checkpoint, "temperature": 1.0},
        "teacher_inner_policy_metadata": {"checkpoint": policy_checkpoint},
        "teacher_value_metadata": {"checkpoint": value_checkpoint, "config_hash": "cfg"},
        "pruner_name": "value_top_k",
        "pruner_version": pruner_version,
        "rng_sampling": "independent",
    }


def _target_metadata(
    *,
    max_depth: int = 4,
    time_budget_ms: float | None = 5000.0,
    pruner_version: str = "1",
) -> dict:
    return {
        "oracle_beam_width": 32,
        "target_beam_width": 8,
        "top_k_actions": 8,
        "max_depth": max_depth,
        "max_continuation_steps": 8,
        "time_budget_ms": time_budget_ms,
        "exhaustive_root_actions": True,
        "pruner_name": "value_top_k",
        "pruner_version": pruner_version,
        "rng_sampling": "independent",
    }


def _record(
    provenance: dict,
    *,
    schema_version: int = ORACLE_RECORD_SCHEMA_VERSION,
    max_depth: int = 4,
    target_pruner_version: str | None = None,
) -> dict:
    pruner_version = (
        provenance["pruner_version"]
        if target_pruner_version is None
        else target_pruner_version
    )
    return {
        "record_type": "combat_oracle_decision",
        "record_schema_version": schema_version,
        "decision_point_id": "d",
        "provenance": provenance,
        "oracle_targets": {
            "metadata": _target_metadata(
                max_depth=max_depth,
                pruner_version=pruner_version,
            )
        },
    }


def _write(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_single_teacher_summary_retains_complete_provenance(tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    provenance = _provenance(value_checkpoint="value-a")
    _write(first, [_record(provenance)])
    _write(second, [_record(dict(reversed(list(provenance.items()))))])

    summary = inspect_oracle_teacher_provenance([first, second])
    payload = summary.to_json()

    assert ORACLE_RECORD_SCHEMA_VERSION == 7
    assert summary.mixed is False
    assert summary.record_count == 2
    assert len(summary.teacher_fingerprints) == 1
    assert payload["schema_version"] == 2
    assert payload["oracle_record_schema_version"] == 7
    assert payload["teachers"][0]["provenance"]["teacher_value_metadata"]["checkpoint"] == "value-a"
    assert payload["teachers"][0]["target_generation"]["max_depth"] == 4
    assert payload["teachers"][0]["target_generation"]["pruner_version"] == "1"
    assert payload["teachers"][0]["target_generation"]["rng_sampling"] == "independent"
    assert payload["teachers"][0]["record_count"] == 2


def test_missing_coverage_policy_class_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "missing-coverage-policy.jsonl"
    provenance = _provenance(value_checkpoint="value-a")
    del provenance["teacher_coverage_policy_class"]
    _write(path, [_record(provenance)])

    with pytest.raises(ValueError, match="teacher_coverage_policy_class"):
        inspect_oracle_teacher_provenance([path])


def test_null_coverage_policy_class_is_allowed(tmp_path: Path) -> None:
    path = tmp_path / "null-coverage-policy.jsonl"
    provenance = _provenance(value_checkpoint="value-a")
    provenance["teacher_coverage_policy_class"] = None
    _write(path, [_record(provenance)])

    summary = inspect_oracle_teacher_provenance([path])

    assert summary.to_json()["teachers"][0]["provenance"]["teacher_coverage_policy_class"] is None


def test_mixed_teacher_is_rejected_by_default_and_explicitly_recorded_when_allowed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mixed.jsonl"
    _write(
        path,
        [
            _record(_provenance(value_checkpoint="value-a")),
            _record(_provenance(value_checkpoint="value-b")),
        ],
    )

    with pytest.raises(ValueError, match="mixed Oracle teacher provenance"):
        inspect_oracle_teacher_provenance([path])

    summary = inspect_oracle_teacher_provenance([path], allow_mixed_teachers=True)

    assert summary.mixed is True
    assert len(summary.teacher_fingerprints) == 2
    assert summary.to_json()["teacher_count"] == 2


def test_target_generation_budget_mismatch_is_treated_as_mixed_teacher(tmp_path: Path) -> None:
    path = tmp_path / "mixed-budget.jsonl"
    provenance = _provenance(value_checkpoint="value-a")
    _write(
        path,
        [
            _record(provenance, max_depth=1),
            _record(provenance, max_depth=4),
        ],
    )

    with pytest.raises(ValueError, match="target-generation config"):
        inspect_oracle_teacher_provenance([path])

    summary = inspect_oracle_teacher_provenance([path], allow_mixed_teachers=True)
    assert summary.mixed is True
    assert {entry.target_generation["max_depth"] for entry in summary.teachers} == {1, 4}


def test_pruner_version_mismatch_is_treated_as_mixed_teacher(tmp_path: Path) -> None:
    path = tmp_path / "mixed-pruner.jsonl"
    _write(
        path,
        [
            _record(_provenance(value_checkpoint="value-a", pruner_version="1")),
            _record(_provenance(value_checkpoint="value-a", pruner_version="2")),
        ],
    )

    with pytest.raises(ValueError, match="target-generation config"):
        inspect_oracle_teacher_provenance([path])

    summary = inspect_oracle_teacher_provenance([path], allow_mixed_teachers=True)
    assert summary.mixed is True
    assert {entry.target_generation["pruner_version"] for entry in summary.teachers} == {
        "1",
        "2",
    }


def test_duplicate_search_identity_fields_must_match_provenance(tmp_path: Path) -> None:
    path = tmp_path / "inconsistent-pruner.jsonl"
    record = _record(
        _provenance(value_checkpoint="value-a", pruner_version="1"),
        target_pruner_version="2",
    )
    _write(path, [record])

    with pytest.raises(ValueError, match="provenance.pruner_version does not match"):
        inspect_oracle_teacher_provenance([path])


def test_missing_target_generation_metadata_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "missing-target-metadata.jsonl"
    record = _record(_provenance(value_checkpoint="value-a"))
    del record["oracle_targets"]
    _write(path, [record])

    with pytest.raises(ValueError, match="oracle_targets"):
        inspect_oracle_teacher_provenance([path])


def test_oracle_current_schema_is_required_for_teacher_provenance_validation(tmp_path: Path) -> None:
    path = tmp_path / "oracle-v6.jsonl"
    _write(path, [_record(_provenance(value_checkpoint="value-a"), schema_version=6)])

    with pytest.raises(ValueError, match="expected Oracle record schema v7"):
        inspect_oracle_teacher_provenance([path])


def test_eval_teacher_set_must_match_artifact_unless_explicitly_overridden(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write(train_path, [_record(_provenance(value_checkpoint="value-a"))])
    _write(eval_path, [_record(_provenance(value_checkpoint="value-b"))])
    training = inspect_oracle_teacher_provenance([train_path]).to_json()
    evaluation = inspect_oracle_teacher_provenance([eval_path])

    assert teacher_provenance_matches(training, evaluation) is False
    with pytest.raises(ValueError, match="does not match"):
        require_matching_teacher_provenance(training, evaluation)
    assert (
        require_matching_teacher_provenance(
            training,
            evaluation,
            allow_teacher_mismatch=True,
        )
        is False
    )


def test_v1_artifact_summary_uses_legacy_provenance_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    provenance = _provenance(value_checkpoint="value-a")
    _write(path, [_record(provenance, max_depth=4)])
    evaluation = inspect_oracle_teacher_provenance([path])
    canonical = json.dumps(
        provenance,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    legacy = {
        "schema_version": 1,
        "oracle_record_schema_version": ORACLE_RECORD_SCHEMA_VERSION,
        "teacher_fingerprints": [hashlib.sha256(canonical.encode("utf-8")).hexdigest()],
    }

    assert teacher_provenance_matches(legacy, evaluation) is True


def test_artifact_without_teacher_provenance_is_rejected_by_default(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    _write(path, [_record(_provenance(value_checkpoint="value-a"))])
    evaluation = inspect_oracle_teacher_provenance([path])

    with pytest.raises(ValueError, match="does not record Oracle teacher provenance"):
        require_matching_teacher_provenance(None, evaluation)
