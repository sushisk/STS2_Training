from __future__ import annotations

from pathlib import Path

import pytest
import train_stable_pruner

from sts2_training.decision.learned_pruner import LinearStableFrontierPruner
from sts2_training.decision.pruner_features import PRUNER_FEATURE_NAMES
from sts2_training.decision.pruner_training_data import (
    PairwisePrunerExample,
    PrunerFrontierTrainingExample,
    PrunerNodeTrainingExample,
)


def _features(node_value: float, signal: float) -> tuple[float, ...]:
    values = [0.0] * len(PRUNER_FEATURE_NAMES)
    values[PRUNER_FEATURE_NAMES.index("node_value")] = node_value
    values[PRUNER_FEATURE_NAMES.index("policy_score")] = signal
    return tuple(values)


def _node(node_id: str, target: float, raw_value: float, signal: float) -> PrunerNodeTrainingExample:
    return PrunerNodeTrainingExample(
        source_path="run.jsonl",
        decision_point_id="d",
        search_id="s",
        prune_step_id="p",
        node_id=node_id,
        features=_features(raw_value, signal),
        target_value=target,
        target_source="terminal",
        target_weight=1.0,
        target_beam_width=1,
        baseline_would_keep=raw_value >= 5.0,
        oracle_kept=True,
    )


def test_pairwise_ranker_learns_signal_that_beats_raw_value() -> None:
    frontiers = [
        PrunerFrontierTrainingExample(
            source_path="run.jsonl",
            decision_point_id="d",
            search_id="s",
            prune_step_id="p",
            target_beam_width=1,
            nodes=(
                _node("good", target=10.0, raw_value=1.0, signal=1.0),
                _node("bad", target=0.0, raw_value=10.0, signal=-1.0),
            ),
        )
    ]
    pairs = train_stable_pruner.build_pairwise_examples(frontiers)

    fitted = train_stable_pruner.fit_pairwise_ranker(
        pairs,
        inverse_regularization=100.0,
        seed=0,
    )
    metrics = train_stable_pruner.evaluate_ranker(
        fitted,
        frontiers,
        min_target_gap=1e-6,
    )

    assert metrics["pairwise_accuracy"] == 1.0
    assert metrics["learned_recall_at_k"] == 1.0
    assert metrics["value_top_k_recall_at_k"] == 0.0
    assert metrics["learned_best_value_regret"] == 0.0
    assert metrics["value_top_k_best_value_regret"] == 10.0


def test_weights_payload_round_trips_into_dependency_free_runtime(tmp_path: Path) -> None:
    pairs = [
        PairwisePrunerExample(
            features=_features(0.0, 2.0),
            label=1,
            weight=1.0,
            prune_step_id="p",
            positive_node_id="a",
            negative_node_id="b",
            target_gap=1.0,
        ),
        PairwisePrunerExample(
            features=_features(0.0, -2.0),
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
    payload = train_stable_pruner.weights_payload(
        fitted,
        metrics={"train": {}},
        training_files=[],
        min_target_gap=1e-6,
        terminal_weight=1.0,
        bootstrap_weight=0.5,
    )
    path = tmp_path / "weights.json"
    import json

    path.write_text(json.dumps(payload), encoding="utf-8")

    pruner = LinearStableFrontierPruner.from_weights_file(path)

    assert pruner.feature_names == PRUNER_FEATURE_NAMES
    assert pruner.artifact_metadata["model_type"] == "pairwise_logistic_linear_pruner"


def test_split_source_files_is_deterministic_and_disjoint() -> None:
    paths = [Path(f"run-{index}.jsonl") for index in range(10)]

    first = train_stable_pruner.split_source_files(
        paths,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=7,
    )
    second = train_stable_pruner.split_source_files(
        paths,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=7,
    )

    assert first == second
    assert len(first.train) == 6
    assert len(first.val) == 2
    assert len(first.test) == 2
    assert set(first.train).isdisjoint(first.val)
    assert set(first.train).isdisjoint(first.test)
    assert set(first.val).isdisjoint(first.test)


def test_split_source_files_rejects_invalid_fractions() -> None:
    with pytest.raises(ValueError):
        train_stable_pruner.split_source_files([], val_fraction=0.5, test_fraction=0.5, seed=0)
