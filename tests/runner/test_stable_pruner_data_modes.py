from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from sts2_training.decision.oracle_log import ORACLE_RECORD_SCHEMA_VERSION
from sts2_training.runner import stable_pruner_learn


def _oracle_record() -> dict:
    return {
        "record_type": "combat_oracle_decision",
        "record_schema_version": ORACLE_RECORD_SCHEMA_VERSION,
        "provenance": {},
    }


def _prepared(tmp_path: Path) -> stable_pruner_learn.PreparedLogs:
    oracle_dir = tmp_path / "oracle"
    oracle_dir.mkdir()
    return stable_pruner_learn.PreparedLogs(
        oracle_dir=oracle_dir,
        rl_files=(),
        staged_to_source={},
        sources=(),
        oracle_records=1,
        rl_records=0,
        other_json_records=0,
        malformed_nonempty_lines=0,
    )


def _fresh_args() -> Namespace:
    return Namespace(
        val_fraction=0.2,
        test_fraction=0.15,
        seed=7,
        inverse_regularization=3.0,
        min_target_gap=0.25,
        terminal_weight=2.0,
        bootstrap_weight=0.3,
        allow_mixed_teachers=False,
    )


def test_data_mode_auto_resolves_split_for_fresh_and_train_for_resume() -> None:
    assert stable_pruner_learn.resolve_data_mode(
        "auto",
        plan=stable_pruner_learn.LearningPlan("supervised", "fresh"),
        weights=None,
        oracle_records=3,
    ) == "auto-split"
    assert stable_pruner_learn.resolve_data_mode(
        "auto",
        plan=stable_pruner_learn.LearningPlan("supervised", "resume"),
        weights=Path("weights.json"),
        oracle_records=3,
    ) == "train"
    assert stable_pruner_learn.resolve_data_mode(
        "auto",
        plan=stable_pruner_learn.LearningPlan("rl", "resume"),
        weights=Path("weights.json"),
        oracle_records=0,
    ) == "train"


def test_validate_requires_existing_supervised_artifact() -> None:
    with pytest.raises(ValueError, match="requires --weights"):
        stable_pruner_learn.resolve_data_mode(
            "validate",
            plan=stable_pruner_learn.LearningPlan("supervised", "resume"),
            weights=None,
            oracle_records=1,
        )
    with pytest.raises(ValueError, match="supervised Oracle evaluation only"):
        stable_pruner_learn.resolve_data_mode(
            "validate",
            plan=stable_pruner_learn.LearningPlan("rl", "resume"),
            weights=Path("weights.json"),
            oracle_records=1,
        )


def test_auto_split_is_rejected_for_resume() -> None:
    with pytest.raises(ValueError, match="only valid for fresh supervised"):
        stable_pruner_learn.resolve_data_mode(
            "auto-split",
            plan=stable_pruner_learn.LearningPlan("supervised", "resume"),
            weights=Path("weights.json"),
            oracle_records=1,
        )


def test_train_mode_disables_internal_validation_split(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(stable_pruner_learn, "_require_tool", lambda path: None)
    command = stable_pruner_learn._supervised_fresh_command(
        _fresh_args(),
        prepared=_prepared(tmp_path),
        output=tmp_path / "weights.json",
        data_mode="train",
    )
    assert command[command.index("--val-fraction") + 1] == "0.0"
    assert command[command.index("--test-fraction") + 1] == "0.0"


def test_auto_split_mode_forwards_requested_fractions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(stable_pruner_learn, "_require_tool", lambda path: None)
    command = stable_pruner_learn._supervised_fresh_command(
        _fresh_args(),
        prepared=_prepared(tmp_path),
        output=tmp_path / "weights.json",
        data_mode="auto-split",
    )
    assert command[command.index("--val-fraction") + 1] == "0.2"
    assert command[command.index("--test-fraction") + 1] == "0.15"


def test_validation_command_uses_held_out_evaluator(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(stable_pruner_learn, "_require_tool", lambda path: None)
    args = Namespace(
        weights=Path("weights.json"),
        min_target_gap=None,
        terminal_weight=None,
        bootstrap_weight=None,
        allow_mixed_teachers=False,
        allow_teacher_mismatch=False,
    )
    command = stable_pruner_learn._supervised_validation_command(
        args, prepared=_prepared(tmp_path)
    )
    joined = " ".join(command)
    assert "eval_stable_pruner.py" in joined
    assert "train_stable_pruner.py" not in joined
    assert command[command.index("--weights") + 1] == "weights.json"


def test_run_validation_writes_report_without_running_update(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "oracle.jsonl"
    source.write_text(json.dumps(_oracle_record()) + "\n", encoding="utf-8")
    weights = tmp_path / "weights.json"
    weights.write_text("{}", encoding="utf-8")
    output = tmp_path / "validation.json"
    called = {"validate": 0, "train": 0}

    def fake_validate(command, *, output: Path):
        called["validate"] += 1
        output.write_text(json.dumps({"pairwise_accuracy": 0.75}), encoding="utf-8")

    def fake_train(command):
        called["train"] += 1

    monkeypatch.setattr(stable_pruner_learn, "_run_validation_command", fake_validate)
    monkeypatch.setattr(stable_pruner_learn, "_run_command", fake_train)
    monkeypatch.setattr(stable_pruner_learn, "_require_tool", lambda path: None)
    args = stable_pruner_learn._parse_args(
        [
            str(source),
            "--data-mode",
            "validate",
            "--weights",
            str(weights),
            "--output",
            str(output),
        ]
    )

    summary = stable_pruner_learn.run_learning(args)

    assert called == {"validate": 1, "train": 0}
    assert summary.data_mode == "validate"
    assert summary.output_kind == "validation_report"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["pairwise_accuracy"] == 0.75
    assert payload["one_line_learning_ingest"]["data_mode"] == "validate"
    assert payload["one_line_learning_ingest"]["updates_model"] is False
