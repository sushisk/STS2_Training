from __future__ import annotations

import hashlib
import json

import pytest
import update_stable_pruner_rl

from rl_v3_record_fixture import record
from sts2_training.decision.learned_pruner import LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION
from sts2_training.decision.pruner_features import (
    PRUNER_FEATURE_NAMES,
    PRUNER_FEATURE_SCHEMA_VERSION,
)
from sts2_training.decision.stable_pruner import STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION


def _load(tmp_path, value):
    path = tmp_path / "trajectory.jsonl"
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return update_stable_pruner_rl.load_update_examples(
        [path],
        expected_artifact_sha256="expected",
        coefficients=[0.0] * len(PRUNER_FEATURE_NAMES),
        scale=[1.0] * len(PRUNER_FEATURE_NAMES),
    )


def test_public_entrypoint_rejects_feature_schema_mismatch(tmp_path):
    value = record()
    value["behavior"]["feature_schema_version"] = PRUNER_FEATURE_SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="feature_schema_version mismatch"):
        _load(tmp_path, value)


def test_public_entrypoint_rejects_tampered_behavior_score(tmp_path):
    value = record()
    value["steps"][0]["behavior_scores"][0] = 1.0
    with pytest.raises(ValueError, match="behavior score does not match"):
        _load(tmp_path, value)


def test_public_entrypoint_rejects_tampered_returned_survivor_order(tmp_path):
    value = record()
    step = value["steps"][0]
    step["frontier_features"].append([0.0] * len(PRUNER_FEATURE_NAMES))
    step["behavior_scores"] = [0.0, 0.0, 0.0]
    step["beam_width"] = 2
    step["sampled_indices"] = [0, 1]
    step["returned_indices"] = [1, 0]
    step["selection_log_probability"] = -1.791759469228055
    with pytest.raises(ValueError, match="returned_indices do not match"):
        _load(tmp_path, value)


def test_public_entrypoint_rejects_step_temperature_mismatch(tmp_path):
    value = record()
    value["steps"][0]["temperature"] = 2.0
    with pytest.raises(ValueError, match="step.temperature does not match behavior.temperature"):
        _load(tmp_path, value)


def test_public_entrypoint_rejects_step_sampler_seed_mismatch(tmp_path):
    value = record()
    value["steps"][0]["sampler_seed"] = 999
    with pytest.raises(ValueError, match="step.sampler_seed does not match behavior.sampler_seed"):
        _load(tmp_path, value)


def test_public_entrypoint_rejects_tampered_log_probability(tmp_path):
    value = record()
    value["steps"][0]["selection_log_probability"] = -0.1
    with pytest.raises(ValueError, match="selection_log_probability does not match"):
        _load(tmp_path, value)


def test_public_entrypoint_rejects_paired_result_delta_mismatch(tmp_path):
    value = record()
    value["paired_result"]["learned_nodes_expanded"] = 13
    with pytest.raises(ValueError, match="nodes_expanded_delta does not match paired_result"):
        _load(tmp_path, value)


def test_public_entrypoint_rejects_resource_evaluator_weight_tamper(tmp_path):
    value = record()
    value["reward"]["resource_hp_weight"] = 0.7
    with pytest.raises(ValueError, match="resource_hp_weight"):
        _load(tmp_path, value)


def test_updated_artifact_hashes_input_and_invalidates_stale_metrics(tmp_path):
    trajectory = tmp_path / "batch.jsonl"
    trajectory.write_bytes(b'{"record_type":"stable_pruner_rl_episode"}\n')
    base = {
        "model_type": "pairwise_logistic_linear_pruner",
        "artifact_schema_version": LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION,
        "stable_prune_node_view_schema_version": STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
        "feature_schema_version": PRUNER_FEATURE_SCHEMA_VERSION,
        "oracle_teacher_provenance": {"teacher_fingerprints": ["abc"]},
        "metrics": {"val": {"pairwise_accuracy": 0.5}},
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
        trajectory_files=[trajectory],
        learning_rate=0.01,
        gradient_clip_norm=5.0,
    )

    update = payload["last_rl_update"]
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
