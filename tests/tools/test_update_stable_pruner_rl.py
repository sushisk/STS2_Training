from __future__ import annotations

import hashlib
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


def _step(*, sampler_seed: int = 17) -> dict:
    width = len(PRUNER_FEATURE_NAMES)
    first = [0.0] * width
    second = [0.0] * width
    first[PRUNER_FEATURE_NAMES.index("node_value")] = 1.0
    return {
        "beam_width": 1,
        "stable_prune_node_view_schema_version": STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
        "feature_schema_version": PRUNER_FEATURE_SCHEMA_VERSION,
        "temperature": 1.0,
        "sampler_seed": sampler_seed,
        "frontier_features": [first, second],
        "behavior_scores": [0.0, 0.0],
        "sampled_indices": [0],
        "returned_indices": [0],
        "selection_log_probability": -0.6931471805599453,
    }


def _behavior(artifact_sha: str, *, sampler_seed: int = 17) -> dict:
    return {
        "artifact_sha256": artifact_sha,
        "artifact_schema_version": LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION,
        "stable_prune_node_view_schema_version": STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
        "feature_schema_version": PRUNER_FEATURE_SCHEMA_VERSION,
        "temperature": 1.0,
        "sampler_seed": sampler_seed,
    }


def _record(artifact_sha: str = "expected") -> dict:
    return {
        "record_type": "stable_pruner_rl_episode",
        "record_schema_version": RL_TRAJECTORY_SCHEMA_VERSION,
        "behavior": _behavior(artifact_sha),
        "reward": {
            "outcome_delta": 1.0,
            "nodes_expanded_delta": 2,
            "beam_ms_delta": 3.0,
            "node_cost_weight": 0.1,
            "beam_ms_cost_weight": 0.2,
            "total": 0.2,
        },
        "paired_result": {
            "baseline_outcome": "defeat",
            "learned_outcome": "victory",
            "baseline_nodes_expanded": 10,
            "learned_nodes_expanded": 12,
            "baseline_beam_total_ms": 4.0,
            "learned_beam_total_ms": 7.0,
        },
        "steps": [_step()],
    }


def _artifact_base() -> dict:
    return {
        "model_type": "pairwise_logistic_linear_pruner",
        "artifact_schema_version": LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION,
        "stable_prune_node_view_schema_version": STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
        "feature_schema_version": PRUNER_FEATURE_SCHEMA_VERSION,
        "oracle_teacher_provenance": {"teacher_fingerprints": ["abc"]},
        "metrics": {"val": {"pairwise_accuracy": 0.5}},
    }


def _load(path: Path, *, expected_sha: str = "expected"):
    return update_stable_pruner_rl.load_update_examples(
        [path],
        expected_artifact_sha256=expected_sha,
        coefficients=[0.0] * len(PRUNER_FEATURE_NAMES),
        scale=[1.0] * len(PRUNER_FEATURE_NAMES),
    )


def _write(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


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
    _write(path, _record("wrong"))
    with pytest.raises(ValueError, match="does not match update base"):
        _load(path)


def test_loader_rejects_behavior_schema_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    record = _record()
    record["behavior"]["feature_schema_version"] = PRUNER_FEATURE_SCHEMA_VERSION + 1
    _write(path, record)
    with pytest.raises(ValueError, match="feature_schema_version mismatch"):
        _load(path)


def test_loader_rejects_tampered_returned_survivor_order(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    record = _record()
    step = record["steps"][0]
    width = len(PRUNER_FEATURE_NAMES)
    third = [0.0] * width
    step["frontier_features"].append(third)
    step["behavior_scores"] = [0.0, 0.0, 0.0]
    step["beam_width"] = 2
    step["sampled_indices"] = [0, 1]
    step["returned_indices"] = [1, 0]
    step["selection_log_probability"] = -1.791759469228055
    _write(path, record)
    with pytest.raises(ValueError, match="returned_indices do not match"):
        _load(path)


def test_loader_rejects_step_temperature_that_differs_from_behavior(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    record = _record()
    record["steps"][0]["temperature"] = 2.0
    _write(path, record)
    with pytest.raises(ValueError, match="step.temperature does not match behavior.temperature"):
        _load(path)


def test_loader_rejects_step_sampler_seed_that_differs_from_behavior(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    record = _record()
    record["steps"][0]["sampler_seed"] = 999
    _write(path, record)
    with pytest.raises(ValueError, match="step.sampler_seed does not match behavior.sampler_seed"):
        _load(path)


def test_loader_rejects_tampered_reward_total(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    record = _record()
    record["reward"]["total"] = 0.3
    _write(path, record)
    with pytest.raises(ValueError, match="reward.total does not match recomputed"):
        _load(path)


def test_loader_rejects_reward_delta_that_disagrees_with_paired_result(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    record = _record()
    record["paired_result"]["learned_nodes_expanded"] = 13
    _write(path, record)
    with pytest.raises(ValueError, match="nodes_expanded_delta does not match paired_result"):
        _load(path)


def test_loader_accepts_fully_consistent_exact_trajectory(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    _write(path, _record())
    examples = _load(path)
    assert len(examples) == 1
    assert examples[0].reward == pytest.approx(0.2)


def test_updated_artifact_hashes_trajectory_bytes_and_invalidates_old_metrics(
    tmp_path: Path,
) -> None:
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
    trajectory = tmp_path / "batch.jsonl"
    trajectory.write_bytes(b'{"record_type":"stable_pruner_rl_episode"}\n')
    base = _artifact_base()

    payload = update_stable_pruner_rl.updated_artifact_payload(
        base,
        base_artifact_sha256="parent",
        coefficients=[0.1],
        result=result,
        trajectory_files=[trajectory],
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
    update = payload["last_rl_update"]
    assert update["parent_artifact_sha256"] == "parent"
    assert (
        update["behavior_contract"]["rl_trajectory_schema_version"]
        == RL_TRAJECTORY_SCHEMA_VERSION
    )
    assert update["trajectory_inputs"] == [
        {
            "path": str(trajectory),
            "sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
            "hash_scope": "updater_input_bytes",
        }
    ]
    assert payload["metrics"] is None
    assert payload["metrics_status"] == {
        "status": "requires_revalidation",
        "update_stage": "rl_resume",
        "parent_artifact_sha256": "parent",
    }
    assert payload["metrics_history"][0]["artifact_sha256"] == "parent"
    assert payload["metrics_history"][0]["metrics"] == base["metrics"]
    assert len(payload["rl_finetuning_history"]) == 1
