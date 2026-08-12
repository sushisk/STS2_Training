from __future__ import annotations

import eval_stable_pruner

from sts2_training.decision.learned_pruner import LinearStableFrontierPruner
from sts2_training.decision.pruner_features import PRUNER_FEATURE_NAMES
from sts2_training.decision.pruner_training_data import (
    PrunerFrontierTrainingExample,
    PrunerNodeTrainingExample,
)


def _features(raw_value: float, signal: float) -> tuple[float, ...]:
    values = [0.0] * len(PRUNER_FEATURE_NAMES)
    values[PRUNER_FEATURE_NAMES.index("node_value")] = raw_value
    values[PRUNER_FEATURE_NAMES.index("policy_score")] = signal
    return tuple(values)


def _node(node_id: str, *, target: float, raw_value: float, signal: float) -> PrunerNodeTrainingExample:
    return PrunerNodeTrainingExample(
        source_path="heldout.jsonl",
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


def test_evaluate_artifact_compares_learned_ranker_to_value_top_k() -> None:
    coefficients = [0.0] * len(PRUNER_FEATURE_NAMES)
    coefficients[PRUNER_FEATURE_NAMES.index("policy_score")] = 1.0
    pruner = LinearStableFrontierPruner(
        feature_names=PRUNER_FEATURE_NAMES,
        coefficients=coefficients,
    )
    frontier = PrunerFrontierTrainingExample(
        source_path="heldout.jsonl",
        decision_point_id="d",
        search_id="s",
        prune_step_id="p",
        target_beam_width=1,
        nodes=(
            _node("good", target=10.0, raw_value=1.0, signal=1.0),
            _node("bad", target=0.0, raw_value=10.0, signal=-1.0),
        ),
    )

    metrics = eval_stable_pruner.evaluate_artifact(pruner, [frontier])

    assert metrics["pairwise_accuracy"] == 1.0
    assert metrics["learned_recall_at_k"] == 1.0
    assert metrics["value_top_k_recall_at_k"] == 0.0
    assert metrics["learned_best_value_regret"] == 0.0
    assert metrics["value_top_k_best_value_regret"] == 10.0
