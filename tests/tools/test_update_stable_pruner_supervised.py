from __future__ import annotations

from pathlib import Path

import pytest
import update_stable_pruner_supervised

from sts2_training.decision.pruner_training_data import PairwisePrunerExample


def _pair(value: float, label: int, *, weight: float = 1.0) -> PairwisePrunerExample:
    return PairwisePrunerExample(
        features=(value,),
        label=label,
        weight=weight,
        prune_step_id="p",
        positive_node_id="a",
        negative_node_id="b",
        target_gap=1.0,
    )


def _provenance(*fingerprints: str) -> dict:
    return {
        "schema_version": 1,
        "oracle_record_schema_version": 3,
        "mixed": len(fingerprints) > 1,
        "teacher_count": len(fingerprints),
        "record_count": len(fingerprints),
        "source_files": [f"{fingerprint}.jsonl" for fingerprint in fingerprints],
        "teacher_fingerprints": list(fingerprints),
        "teachers": [],
    }


def test_supervised_resume_moves_weight_toward_pairwise_signal() -> None:
    updated, result = update_stable_pruner_supervised.apply_supervised_resume(
        coefficients=[0.0],
        scale=[1.0],
        pairs=[_pair(1.0, 1), _pair(-1.0, 0)],
        learning_rate=0.5,
        epochs=2,
        gradient_clip_norm=10.0,
        inverse_regularization=1.0,
    )

    assert updated[0] > 0.0
    assert result.examples == 2
    assert result.epochs == 2
    assert result.mean_weighted_log_loss_after < result.mean_weighted_log_loss_before
    assert result.regularized_objective_after < result.regularized_objective_before
    assert result.coefficient_delta_norm > 0.0


def test_supervised_resume_respects_pair_weights() -> None:
    updated, _ = update_stable_pruner_supervised.apply_supervised_resume(
        coefficients=[0.0],
        scale=[1.0],
        pairs=[_pair(1.0, 1, weight=10.0), _pair(1.0, 0, weight=1.0)],
        learning_rate=0.1,
        epochs=1,
        gradient_clip_norm=10.0,
        inverse_regularization=1.0,
    )

    assert updated[0] > 0.0


def test_supervised_resume_includes_l2_gradient_from_initial_objective() -> None:
    updated, _ = update_stable_pruner_supervised.apply_supervised_resume(
        coefficients=[1.0],
        scale=[1.0],
        pairs=[_pair(0.0, 1), _pair(0.0, 0)],
        learning_rate=0.1,
        epochs=1,
        gradient_clip_norm=10.0,
        inverse_regularization=1.0,
    )

    # S=2, C=1 => L2 gradient is w/(S*C)=0.5, while zero features
    # contribute no data gradient.
    assert updated[0] == pytest.approx(0.95)


def test_resume_config_defaults_to_artifact_training_values() -> None:
    base = {
        "training": {
            "min_target_gap": 0.25,
            "terminal_weight": 1.5,
            "bootstrap_weight": 0.2,
            "inverse_regularization": 2.5,
        }
    }

    assert update_stable_pruner_supervised.resolve_resume_config(
        base,
        min_target_gap=None,
        terminal_weight=None,
        bootstrap_weight=None,
    ) == (0.25, 1.5, 0.2)
    assert update_stable_pruner_supervised.resolve_resume_config(
        base,
        min_target_gap=0.5,
        terminal_weight=2.0,
        bootstrap_weight=0.4,
    ) == (0.5, 2.0, 0.4)
    assert update_stable_pruner_supervised.resolve_inverse_regularization(
        base, override=None
    ) == 2.5
    assert update_stable_pruner_supervised.resolve_inverse_regularization(
        base, override=4.0
    ) == 4.0


def test_updated_artifact_unions_teacher_provenance_and_invalidates_old_metrics() -> None:
    base = {
        "model_type": "pairwise_logistic_linear_pruner",
        "training": {"inverse_regularization": 2.0},
        "oracle_teacher_provenance": _provenance("aaa"),
        "oracle_dataset_provenance": _provenance("aaa"),
        "metrics": {"val": {"pairwise_accuracy": 0.75}},
        "rl_finetuning_history": [{"parent_artifact_sha256": "older"}],
    }
    result = update_stable_pruner_supervised.SupervisedResumeResult(
        examples=2,
        epochs=1,
        mean_weighted_log_loss_before=0.7,
        mean_weighted_log_loss_after=0.6,
        gradient_norm_last_epoch=0.5,
        coefficient_delta_norm=0.1,
        regularized_objective_before=0.8,
        regularized_objective_after=0.7,
    )

    payload = update_stable_pruner_supervised.updated_artifact_payload(
        base,
        base_artifact_sha256="parent",
        coefficients=[0.1],
        result=result,
        source_files=[Path("new.jsonl")],
        teacher_provenance=_provenance("bbb"),
        learning_rate=0.01,
        epochs=1,
        gradient_clip_norm=5.0,
        min_target_gap=1e-6,
        terminal_weight=1.0,
        bootstrap_weight=0.5,
        inverse_regularization=2.0,
        teacher_provenance_matched=False,
    )

    assert payload["oracle_teacher_provenance"]["teacher_fingerprints"] == ["aaa", "bbb"]
    assert payload["oracle_teacher_provenance"]["mixed"] is True
    assert payload["oracle_dataset_provenance"]["teacher_fingerprints"] == ["aaa", "bbb"]
    assert payload["rl_finetuning_history"] == [{"parent_artifact_sha256": "older"}]
    assert payload["last_supervised_update"]["parent_artifact_sha256"] == "parent"
    assert payload["last_supervised_update"]["teacher_provenance_matched_before_update"] is False
    assert payload["current_training_contract"]["inverse_regularization"] == 2.0
    assert payload["last_supervised_update"]["algorithm"].endswith("l2_full_batch_gradient")
    assert payload["metrics"] is None
    assert payload["metrics_status"]["status"] == "requires_revalidation"
    assert payload["metrics_history"][0]["artifact_sha256"] == "parent"
    assert payload["metrics_history"][0]["metrics"] == base["metrics"]
    assert len(payload["supervised_finetuning_history"]) == 1


@pytest.mark.parametrize("epochs", [0, -1])
def test_supervised_resume_rejects_invalid_epochs(epochs: int) -> None:
    with pytest.raises(ValueError, match="epochs"):
        update_stable_pruner_supervised.apply_supervised_resume(
            coefficients=[0.0],
            scale=[1.0],
            pairs=[_pair(1.0, 1)],
            learning_rate=0.1,
            epochs=epochs,
            gradient_clip_norm=1.0,
            inverse_regularization=1.0,
        )
