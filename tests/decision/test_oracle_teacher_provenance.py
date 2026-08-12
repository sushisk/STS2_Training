from __future__ import annotations

import json
from pathlib import Path

import pytest

from sts2_training.decision.oracle_log import ORACLE_RECORD_SCHEMA_VERSION
from sts2_training.decision.oracle_teacher_provenance import (
    inspect_oracle_teacher_provenance,
    require_matching_teacher_provenance,
    teacher_provenance_matches,
)


def _provenance(*, value_checkpoint: str, policy_checkpoint: str = "policy-a") -> dict:
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
        "pruner_version": "1",
        "rng_sampling": "independent",
    }


def _record(provenance: dict, *, schema_version: int = ORACLE_RECORD_SCHEMA_VERSION) -> dict:
    return {
        "record_type": "combat_oracle_decision",
        "record_schema_version": schema_version,
        "decision_point_id": "d",
        "provenance": provenance,
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

    assert summary.mixed is False
    assert summary.record_count == 2
    assert len(summary.teacher_fingerprints) == 1
    assert payload["oracle_record_schema_version"] == ORACLE_RECORD_SCHEMA_VERSION
    assert payload["teachers"][0]["provenance"]["teacher_value_metadata"]["checkpoint"] == "value-a"
    assert payload["teachers"][0]["record_count"] == 2


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


def test_oracle_v3_is_required_for_teacher_provenance_validation(tmp_path: Path) -> None:
    path = tmp_path / "old.jsonl"
    _write(path, [_record(_provenance(value_checkpoint="value-a"), schema_version=2)])

    with pytest.raises(ValueError, match="expected Oracle record schema"):
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


def test_artifact_without_teacher_provenance_is_rejected_by_default(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    _write(path, [_record(_provenance(value_checkpoint="value-a"))])
    evaluation = inspect_oracle_teacher_provenance([path])

    with pytest.raises(ValueError, match="does not record Oracle teacher provenance"):
        require_matching_teacher_provenance(None, evaluation)
