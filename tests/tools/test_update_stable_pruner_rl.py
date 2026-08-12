from __future__ import annotations

import json
from pathlib import Path

import pytest
import update_stable_pruner_rl

from sts2_training.decision.learned_pruner import LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION
from sts2_training.decision.pruner_features import (
    PRUNER_FEATURE_NAMES,
    PRUNER_FEATURE_SCHEMA_VERSION,
)
from sts2_training.decision.stable_pruner import STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION
from sts2_training.runner.stable_pruner_rl import RL_TRAJECTORY_SCHEMA_VERSION


def _step() -> dict:
    width = len(PRUNER_FEATURE_NAMES)
    first = [0.0] * width
    second = [0.0] * width
    first[PRUNER_FEATURE_NAMES.index("node_value")] = 1.0
    return {
        "beam_width": 1,
        "stable_prune_node_view_schema_version": STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
        "feature_schema_version": PRUNER_FEATURE_SCHEMA_VERSION,
        "temperature": 1.0,
        "frontier_features": [first, second],
        "behavior_scores": [0.0, 0.0],
        "sampled_indices": [0],
        "returned_indices": [0],
        "selection_log_probability": -0.6931471805599453,
    }


def _behavior(artifact_sha: str) -> dict:
    return {
        "artifact_sha256": artifact_sha,
        "artifact_schema_version": LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION,
        "stable_prune_node_view_schema_version": STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
        "feature_schema_version": PRUNER_FEATURE_SCHEMA_VERSION,
    }


def _artifact_base() -> dict:
    return {
        "model_type": "pairwise_logistic_linear_pruner",
        "artifact_schema_version": LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION,
        "stable_prune_node_view_schema_version": STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
        "feature_schema_version": PRUNER_FEATURE_SCHEMA_VERSION,
        "oracle_teacher_provenance": {"teacher_fingerprints": ["abc"]},
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
                "behavior": _behavior("wrong"),
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


def test_loader_rejects_behavior_schema_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    behavior = _behavior("expected")
    behavior["feature_schema_version"] = PRUNER_FEATURE_SCHEMA_VERSION + 1
    path.write_text(
        json.dumps(
            {
                "record_type": "stable_pruner_rl_episode",
                "record_schema_version": RL_TRAJECTORY_SCHEMA_VERSION,
                "behavior": behavior,
                "reward": {"total": 1.0},
                "steps": [_step()],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="feature_schema_version mismatch"):
        update_stable_pruner_rl.load_update_examples(
            [path],
            expected_artifact_sha256="expected",
            coefficients=[0.0] * len(PRUNER_FEATURE_NAMES),
            scale=[1.0] * len(PRUNER_FEATURE_NAMES),
        )


def test_loader_rejects_tampered_returned_survivor_order(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    step = _step()
    # Make both nodes selected so returned order is meaningful, but keep frontier > K by
    # adding a third feature row and setting K=2.
    width = len(PRUNER_FEATURE_NAMES)
    third = [0.0] * width
    step["frontier_features"].append(third)
    step["behavior_scores"] = [0.0, 0.0, 0.0]
    step["beam_width"] = 2
    step["sampled_indices"] = [0, 1]
    step["returned_indices"] = [1, 0]
    step["selection_log_probability"] = -1.791759469228055
    path.write_text(
        json.dumps(
            {
                "record_type": "stable_pruner_rl_episode",
                "record_schema_version": RL_TRAJECTORY_SCHEMA_VERSION,
                "behavior": _behavior("expected"),
                "reward": {"total": 1.0},
                "steps": [step],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="returned_indices do not match"):
        update_stable_pruner_rl.load_update_examples(
            [path],
            expected_artifact_sha256="expected",
            coefficients=[0.0] * len(PRUNER_FEATURE_NAMES),
            scale=[1.0] * len(PRUNER_FEATURE_NAMES),
        )


def test_updated_artifact_preserves_teacher_provenance_and_appends_history() -> None:
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
        _artifact_base(),
        base_artifact_sha256="parent",
        coefficients=[0.1],
        result=result,
        trajectory_files=[Path("batch.jsonl")],
        learning_rate=0.01,
        gradient_clip_norm=5.0,
    )

    assert payload["oracle_teacher_provenance"] == {"teacher_fingerprints": ["abc"]}
    assert payload["artifact_schema_version"] == LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION
    assert (
        payload["stable_prune_node_view_schema_version"]
        == STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION
    )
    assert payload["feature_schema_version"] == PRUNER_FEATURE_SCHEMA_VERSION
    assert payload["last_rl_update"]["parent_artifact_sha256"] == "parent"
    assert (
        payload["last_rl_update"]["behavior_contract"]["rl_trajectory_schema_version"]
        == RL_TRAJECTORY_SCHEMA_VERSION
    )
    assert len(payload["rl_finetuning_history"]) == 1
