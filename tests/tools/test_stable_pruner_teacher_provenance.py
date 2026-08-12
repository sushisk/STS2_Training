from __future__ import annotations

import json
from pathlib import Path

import train_stable_pruner

from sts2_training.decision.learned_pruner import LinearStableFrontierPruner
from sts2_training.decision.pruner_features import PRUNER_FEATURE_NAMES
from sts2_training.decision.pruner_training_data import PairwisePrunerExample


def _features(signal: float) -> tuple[float, ...]:
    values = [0.0] * len(PRUNER_FEATURE_NAMES)
    values[PRUNER_FEATURE_NAMES.index("policy_score")] = signal
    return tuple(values)


def test_artifact_retains_training_and_dataset_teacher_provenance(tmp_path: Path) -> None:
    pairs = [
        PairwisePrunerExample(
            features=_features(1.0),
            label=1,
            weight=1.0,
            prune_step_id="p",
            positive_node_id="a",
            negative_node_id="b",
            target_gap=1.0,
        ),
        PairwisePrunerExample(
            features=_features(-1.0),
            label=0,
            weight=1.0,
            prune_step_id="p",
            positive_node_id="b",
            negative_node_id="a",
            target_gap=1.0,
        ),
    ]
    fitted = train_stable_pruner.fit_pairwise_ranker(
        pairs,
        inverse_regularization=10.0,
        seed=0,
    )
    training_provenance = {
        "oracle_record_schema_version": 3,
        "teacher_fingerprints": ["teacher-a"],
        "teachers": [{"fingerprint_sha256": "teacher-a", "provenance": {"x": 1}}],
    }
    dataset_provenance = {
        "oracle_record_schema_version": 3,
        "teacher_fingerprints": ["teacher-a", "teacher-b"],
        "teachers": [],
    }
    payload = train_stable_pruner.weights_payload(
        fitted,
        metrics={"train": {}},
        training_files=[],
        min_target_gap=1e-6,
        terminal_weight=1.0,
        bootstrap_weight=0.5,
        oracle_teacher_provenance=training_provenance,
        oracle_dataset_provenance=dataset_provenance,
    )
    path = tmp_path / "weights.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    pruner = LinearStableFrontierPruner.from_weights_file(path)

    assert pruner.artifact_metadata["oracle_teacher_provenance"] == training_provenance
    assert pruner.artifact_metadata["oracle_dataset_provenance"] == dataset_provenance
