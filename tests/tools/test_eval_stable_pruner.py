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


def _node(
    node_id: str,
    *,
    target: float | None,
    raw_value: float,
    signal: float,
) -> PrunerNodeTrainingExample:
    labeled = target is not None
    return PrunerNodeTrainingExample(
        source_path="heldout.jsonl",
        decision_point_id="d",
        search_id="s",
        prune_step_id="p",
        node_id=node_id,
        features=_features(raw_value, signal),
        target_value=target,
        target_source="terminal" if labeled else "no_target",
        target_weight=1.0 if labeled else None,
        target_beam_width=1,
        baseline_would_keep=raw_value >= 5.0,
        oracle_kept=labeled,
        censored=not labeled,
        censor_reason=None if labeled else "oracle_pruned_before_followup",
    )


def _policy_score_pruner(*, metadata: dict | None = None) -> LinearStableFrontierPruner:
    coefficients = [0.0] * len(PRUNER_FEATURE_NAMES)
    coefficients[PRUNER_FEATURE_NAMES.index("policy_score")] = 1.0
    return LinearStableFrontierPruner(
        feature_names=PRUNER_FEATURE_NAMES,
        coefficients=coefficients,
        artifact_metadata=metadata,
    )


def test_evaluate_artifact_compares_learned_ranker_to_value_top_k() -> None:
    pruner = _policy_score_pruner()
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
    assert metrics["label_coverage"] == 1.0
    assert metrics["learned_selected_no_target_rate"] == 0.0


def test_evaluate_artifact_ranks_no_target_nodes_in_runtime_frontier() -> None:
    pruner = _policy_score_pruner()
    frontier = PrunerFrontierTrainingExample(
        source_path="heldout.jsonl",
        decision_point_id="d",
        search_id="s",
        prune_step_id="p",
        target_beam_width=1,
        nodes=(
            _node("unknown", target=None, raw_value=0.0, signal=3.0),
            _node("good", target=10.0, raw_value=1.0, signal=2.0),
            _node("bad", target=0.0, raw_value=10.0, signal=1.0),
        ),
    )

    metrics = eval_stable_pruner.evaluate_artifact(pruner, [frontier])

    assert metrics["label_coverage"] == 2 / 3
    assert metrics["learned_selected_no_target_rate"] == 1.0
    assert metrics["learned_recall_at_k"] == 0.0
    assert metrics["learned_quality_evaluable_frontiers"] == 0
    assert metrics["learned_best_value_regret"] is None
    assert metrics["value_top_k_selected_no_target_rate"] == 0.0


def test_evaluation_config_uses_artifact_training_defaults_and_explicit_overrides() -> None:
    pruner = _policy_score_pruner(
        metadata={
            "training": {
                "min_target_gap": 0.25,
                "terminal_weight": 1.5,
                "bootstrap_weight": 0.2,
            }
        }
    )

    config = eval_stable_pruner.resolve_evaluation_config(
        pruner,
        min_target_gap=None,
        terminal_weight=None,
        bootstrap_weight=None,
    )

    assert config["min_target_gap"] == 0.25
    assert config["terminal_weight"] == 1.5
    assert config["bootstrap_weight"] == 0.2
    assert config["sources"] == {
        "min_target_gap": "artifact_training",
        "terminal_weight": "artifact_training",
        "bootstrap_weight": "artifact_training",
    }

    overridden = eval_stable_pruner.resolve_evaluation_config(
        pruner,
        min_target_gap=0.5,
        terminal_weight=None,
        bootstrap_weight=None,
    )
    assert overridden["min_target_gap"] == 0.5
    assert overridden["sources"]["min_target_gap"] == "cli_override"
