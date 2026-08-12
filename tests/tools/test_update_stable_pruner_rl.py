from __future__ import annotations

import json
from pathlib import Path

import pytest
import update_stable_pruner_rl

from sts2_training.decision.pruner_features import PRUNER_FEATURE_NAMES
from sts2_training.runner.stable_pruner_rl import RL_TRAJECTORY_SCHEMA_VERSION


def _step() -> dict:
    width = len(PRUNER_FEATURE_NAMES)
    first = [0.0] * width
    second = [0.0] * width
    first[PRUNER_FEATURE_NAMES.index("node_value")] = 1.0
    return {
        "feature_schema_version": 1,
        "temperature": 1.0,
        "frontier_features": [first, second],
        "behavior_scores": [0.0, 0.0],
        "sampled_indices": [0],
        "selection_log_probability": -0.6931471805599453,
    }


def test_reinforce_update_moves_coefficient_toward_positive_reward_choice() -> None:
    width = len(PRUNER_FEATURE_NAMES)
    coefficients = [0.0] * width
    scale = [1.0] * width
    example = update_stable_pruner_rl.RLUpdateExample(
        reward=1.0,
        steps=(_step(),),
        source_path="trajectory.jsonl",
        line_number=1,
    )

    updated, result = update_stable_pruner_rl.apply_reinforce_update(
        coefficients,
        scale,
        [example],
        learning_rate=0.1,
        gradient_clip_norm=10.0,
    )

    node_value_index = PRUNER_FEATURE_NAMES.index("node_value")
    assert updated[node_value_index] > 0.0
    assert result.examples == 1
    assert result.stochastic_steps == 1
    assert result.coefficient_delta_norm > 0.0


def test_loader_rejects_trajectory_from_different_behavior_artifact(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    path.write_text(
        json.dumps(
            {
                "record_type": "stable_pruner_rl_episode",
                "record_schema_version": RL_TRAJECTORY_SCHEMA_VERSION,
                "behavior": {"artifact_sha256": "wrong"},
                "reward": {"total": 1.0},
                "steps": [_step()],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match update base"):
        update_stable_pruner_rl.load_update_examples(
            [path],
            expected_artifact_sha256="expected",
            coefficients=[0.0] * len(PRUNER_FEATURE_NAMES),
            scale=[1.0] * len(PRUNER_FEATURE_NAMES),
        )


def test_updated_artifact_preserves_teacher_provenance_and_appends_history() -> None:
    base = {
        "model_type": "pairwise_logistic_linear_pruner",
        "oracle_teacher_provenance": {"teacher_fingerprints": ["abc"]},
    }
    result = update_stable_pruner_rl.RLUpdateResult(
        examples=1,
        stochastic_steps=1,
        mean_reward=1.0,
        positive_reward_examples=1,
        negative_reward_examples=0,
        zero_reward_examples=0,
        raw_gradient_norm=1.0,
        applied_gradient_norm=1.0,
        coefficient_delta_norm=0.1,
    )

    payload = update_stable_pruner_rl.updated_artifact_payload(
        base,
        base_artifact_sha256="parent",
        coefficients=[0.1],
        result=result,
        trajectory_files=[Path("batch.jsonl")],
        learning_rate=0.01,
        gradient_clip_norm=5.0,
    )

    assert payload["oracle_teacher_provenance"] == {"teacher_fingerprints": ["abc"]}
    assert payload["last_rl_update"]["parent_artifact_sha256"] == "parent"
    assert len(payload["rl_finetuning_history"]) == 1
