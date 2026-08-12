from __future__ import annotations

import json
from pathlib import Path

import pytest

from sts2_training.decision.oracle_log import ORACLE_RECORD_SCHEMA_VERSION
from sts2_training.runner import stable_pruner_learn


def _oracle_log(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "record_type": "combat_oracle_decision",
                "record_schema_version": ORACLE_RECORD_SCHEMA_VERSION,
                "provenance": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_one_line_resume_rejects_output_equal_to_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _oracle_log(tmp_path / "oracle.jsonl")
    weights = tmp_path / "weights.json"
    weights.write_text("{}", encoding="utf-8")
    called = False

    def fail_if_run(command) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(stable_pruner_learn, "_run_command", fail_if_run)
    args = stable_pruner_learn._parse_args(
        [
            str(source),
            "--learn",
            "supervised",
            "--start",
            "resume",
            "--data-mode",
            "train",
            "--weights",
            str(weights),
            "--output",
            str(weights),
        ]
    )

    with pytest.raises(ValueError, match="--output must differ from --weights"):
        stable_pruner_learn.run_learning(args)

    assert not called


def test_validation_rejects_report_output_equal_to_model_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _oracle_log(tmp_path / "validation.jsonl")
    weights = tmp_path / "weights.json"
    weights.write_text("{}", encoding="utf-8")
    called = False

    def fail_if_validation_runs(command, *, output) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(
        stable_pruner_learn, "_run_validation_command", fail_if_validation_runs
    )
    args = stable_pruner_learn._parse_args(
        [
            str(source),
            "--data-mode",
            "validate",
            "--weights",
            str(weights),
            "--output",
            str(weights),
        ]
    )

    with pytest.raises(ValueError, match="--output must differ from --weights"):
        stable_pruner_learn.run_learning(args)

    assert not called
