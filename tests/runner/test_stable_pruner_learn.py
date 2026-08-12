from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

from sts2_training.decision.oracle_log import ORACLE_RECORD_SCHEMA_VERSION
from sts2_training.runner import stable_pruner_learn
from sts2_training.runner.stable_pruner_rl import RL_TRAJECTORY_SCHEMA_VERSION


def _oracle_record() -> dict:
    return {
        "record_type": "combat_oracle_decision",
        "record_schema_version": ORACLE_RECORD_SCHEMA_VERSION,
        "provenance": {},
    }


def _rl_record() -> dict:
    return {
        "record_type": "stable_pruner_rl_episode",
        "record_schema_version": RL_TRAJECTORY_SCHEMA_VERSION,
        "behavior": {"artifact_sha256": "abc"},
    }


def _prepared(tmp_path: Path, *, oracle: int = 1, rl: int = 0):
    return stable_pruner_learn.PreparedLogs(
        oracle_dir=tmp_path / "oracle",
        rl_files=tuple(tmp_path / f"rl-{index}.jsonl" for index in range(rl)),
        staged_to_source={},
        sources=(),
        oracle_records=oracle,
        rl_records=rl,
        other_json_records=0,
        malformed_nonempty_lines=0,
    )


def test_prepare_logs_accepts_current_jsonl_and_prefixed_log_lines(tmp_path: Path) -> None:
    source = tmp_path / "runtime.log"
    source.write_text(
        json.dumps(_oracle_record())
        + "\n"
        + "2026-08-12 INFO "
        + json.dumps(_rl_record())
        + " trailing-text\n"
        + json.dumps({"record_type": "selection_audit"})
        + "\n"
        + "not json\n",
        encoding="utf-8",
    )

    prepared = stable_pruner_learn.prepare_logs(
        [str(source)], staging_root=tmp_path / "staging"
    )

    assert prepared.oracle_records == 1
    assert prepared.rl_records == 1
    assert prepared.other_json_records == 1
    assert prepared.malformed_nonempty_lines == 1
    assert len(prepared.rl_files) == 1
    oracle_staged = next(prepared.oracle_dir.glob("*.jsonl"))
    assert json.loads(oracle_staged.read_text(encoding="utf-8"))["record_type"] == (
        "combat_oracle_decision"
    )
    assert prepared.sources[0].sha256 == hashlib.sha256(source.read_bytes()).hexdigest()


def test_discover_log_files_recurses_and_deduplicates(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    first = tmp_path / "a.jsonl"
    second = nested / "b.log"
    ignored = nested / "c.bin"
    first.write_text("{}\n", encoding="utf-8")
    second.write_text("{}\n", encoding="utf-8")
    ignored.write_bytes(b"{}\n")

    discovered = stable_pruner_learn.discover_log_files([str(tmp_path), str(first)])

    assert discovered == tuple(sorted((first.resolve(), second.resolve()), key=str))


def test_learning_plan_supports_fresh_and_resume() -> None:
    assert stable_pruner_learn.resolve_learning_plan(
        "auto", "auto", weights=None, oracle_records=2, rl_records=1
    ) == stable_pruner_learn.LearningPlan("supervised", "fresh")
    assert stable_pruner_learn.resolve_learning_plan(
        "supervised",
        "resume",
        weights=Path("weights.json"),
        oracle_records=2,
        rl_records=0,
    ) == stable_pruner_learn.LearningPlan("supervised", "resume")
    assert stable_pruner_learn.resolve_learning_plan(
        "rl",
        "resume",
        weights=Path("weights.json"),
        oracle_records=0,
        rl_records=2,
    ) == stable_pruner_learn.LearningPlan("rl", "resume")


def test_auto_resume_requires_explicit_learning_type_when_logs_are_mixed() -> None:
    with pytest.raises(ValueError, match="specify --learn"):
        stable_pruner_learn.resolve_learning_plan(
            "auto",
            "auto",
            weights=Path("weights.json"),
            oracle_records=2,
            rl_records=2,
        )


def test_rl_fresh_is_rejected_because_trajectory_is_artifact_bound() -> None:
    with pytest.raises(ValueError, match="cannot start fresh"):
        stable_pruner_learn.resolve_learning_plan(
            "rl", "fresh", weights=None, oracle_records=0, rl_records=2
        )


def test_resume_requires_weights_and_fresh_rejects_weights() -> None:
    with pytest.raises(ValueError, match="resume requires --weights"):
        stable_pruner_learn.resolve_learning_plan(
            "supervised", "resume", weights=None, oracle_records=1, rl_records=0
        )
    with pytest.raises(ValueError, match="must not receive --weights"):
        stable_pruner_learn.resolve_learning_plan(
            "supervised",
            "fresh",
            weights=Path("weights.json"),
            oracle_records=1,
            rl_records=0,
        )


def test_supervised_fresh_command_forwards_training_options(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(stable_pruner_learn, "_require_tool", lambda path: None)
    args = Namespace(
        val_fraction=0.2,
        test_fraction=0.15,
        seed=7,
        inverse_regularization=3.0,
        min_target_gap=0.25,
        terminal_weight=2.0,
        bootstrap_weight=0.3,
        allow_mixed_teachers=True,
    )

    command = stable_pruner_learn._supervised_fresh_command(
        args, prepared=_prepared(tmp_path), output=tmp_path / "weights.json"
    )

    assert "--allow-mixed-teachers" in command
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--min-target-gap") + 1] == "0.25"


def test_supervised_resume_command_uses_resume_updater_and_artifact_defaults(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(stable_pruner_learn, "_require_tool", lambda path: None)
    args = Namespace(
        weights=Path("base.json"),
        learning_rate=0.02,
        epochs=3,
        gradient_clip_norm=4.0,
        min_target_gap=None,
        terminal_weight=None,
        bootstrap_weight=None,
        allow_mixed_teachers=False,
        allow_teacher_mismatch=False,
    )

    command = stable_pruner_learn._supervised_resume_command(
        args, prepared=_prepared(tmp_path), output=tmp_path / "continued.json"
    )

    assert "update_stable_pruner_supervised.py" in " ".join(command)
    assert command[command.index("--weights") + 1] == "base.json"
    assert command[command.index("--epochs") + 1] == "3"
    assert "--min-target-gap" not in command
    assert "--terminal-weight" not in command
    assert "--bootstrap-weight" not in command


def test_rewrite_artifact_restores_original_paths_and_records_learning_plan(tmp_path: Path) -> None:
    original = tmp_path / "source.jsonl"
    original.write_text(json.dumps(_oracle_record()) + "\n", encoding="utf-8")
    staged = tmp_path / "stage" / "000.jsonl"
    staged.parent.mkdir()
    staged.write_text(original.read_text(encoding="utf-8"), encoding="utf-8")
    output = tmp_path / "weights.json"
    output.write_text(
        json.dumps(
            {
                "training": {"source_files": [str(staged)]},
                "oracle_teacher_provenance": {
                    "source_files": [str(staged)],
                    "teachers": [{"source_files": [str(staged)]}],
                },
            }
        ),
        encoding="utf-8",
    )
    source_display = stable_pruner_learn._display_path(original)
    prepared = stable_pruner_learn.PreparedLogs(
        oracle_dir=staged.parent,
        rl_files=(),
        staged_to_source={str(staged): source_display},
        sources=(
            stable_pruner_learn.SourceLogSummary(
                path=source_display,
                sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                oracle_records=1,
                rl_records=0,
                other_json_records=0,
                malformed_nonempty_lines=0,
            ),
        ),
        oracle_records=1,
        rl_records=0,
        other_json_records=0,
        malformed_nonempty_lines=0,
    )

    stable_pruner_learn._rewrite_artifact_provenance(
        output,
        plan=stable_pruner_learn.LearningPlan("supervised", "fresh"),
        prepared=prepared,
        base_weights=None,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["training"]["source_files"] == [source_display]
    assert payload["oracle_teacher_provenance"]["source_files"] == [source_display]
    ingest = payload["one_line_learning_ingest"]
    assert ingest["learn"] == "supervised"
    assert ingest["start"] == "fresh"
    assert ingest["source_logs"][0]["path"] == source_display
    assert ingest["record_counts"]["combat_oracle_decision"] == 1


def test_run_learning_is_one_entrypoint_for_supervised_fresh(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "oracle.jsonl"
    source.write_text(json.dumps(_oracle_record()) + "\n", encoding="utf-8")
    output = tmp_path / "artifact.json"
    captured: list[str] = []

    def fake_run(command):
        captured.extend(command)
        output.write_text(json.dumps({"training": {"source_files": []}}))

    monkeypatch.setattr(stable_pruner_learn, "_run_command", fake_run)
    monkeypatch.setattr(stable_pruner_learn, "_require_tool", lambda path: None)
    args = stable_pruner_learn._parse_args([str(source), "--output", str(output)])

    summary = stable_pruner_learn.run_learning(args)

    assert summary.learn == "supervised"
    assert summary.start == "fresh"
    assert summary.oracle_records == 1
    assert summary.output == str(output)
    assert "train_stable_pruner.py" in " ".join(captured)
    assert output.exists()
