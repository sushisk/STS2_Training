from __future__ import annotations

import json
from pathlib import Path

import pytest

from sts2_training.board_eval.card_features import CardFeatureExtractor
from sts2_training.board_eval.linear_value_function import LinearRunStateValueModel
from sts2_training.board_eval.run_state_value import RunStateValueModel
from sts2_training.board_eval.training_data import MODEL_FEATURE_NAMES
from sts2_training.decision.value import ValueModel


def _extractor() -> CardFeatureExtractor:
    return CardFeatureExtractor(
        [
            {
                "card_id": "STRIKE_IRONCLAD",
                "card_type": "Attack",
                "target_type": "AnyEnemy",
                "energy_cost": "1",
                "damage": "6",
            }
        ]
    )


def _model(**overrides: object) -> LinearRunStateValueModel:
    kwargs: dict[str, object] = {
        "feature_names": MODEL_FEATURE_NAMES,
        "coefficients": [0.0] * len(MODEL_FEATURE_NAMES),
        "intercept": 0.0,
        "extractor": _extractor(),
    }
    kwargs.update(overrides)
    return LinearRunStateValueModel(**kwargs)  # type: ignore[arg-type]


def test_run_state_model_is_separate_from_combat_value_model() -> None:
    model = _model()

    assert isinstance(model, RunStateValueModel)
    assert not isinstance(model, ValueModel)


def test_zero_weights_predict_half_probability() -> None:
    model = _model()

    probability = model.predict_win_probability(
        {"hp": 50, "maxHp": 80, "deck": [{"id": "STRIKE_IRONCLAD"}]}
    )

    assert probability == pytest.approx(0.5)


def test_missing_flag_is_available_to_inference() -> None:
    coefficients = [0.0] * len(MODEL_FEATURE_NAMES)
    coefficients[MODEL_FEATURE_NAMES.index("gold_missing")] = 2.0
    model = _model(coefficients=coefficients)

    assert model.predict_win_probability({}) > model.predict_win_probability({"gold": 0})


def test_unknown_cards_are_counted_when_skip_policy_is_used() -> None:
    coefficients = [0.0] * len(MODEL_FEATURE_NAMES)
    coefficients[MODEL_FEATURE_NAMES.index("deck_unknown_card_count")] = 1.0
    model = _model(coefficients=coefficients)

    known = model.predict_win_probability({"deck": [{"id": "STRIKE_IRONCLAD"}]})
    with_unknown = model.predict_win_probability(
        {"deck": [{"id": "STRIKE_IRONCLAD"}, {"id": "UNKNOWN"}]}
    )

    assert with_unknown > known


def test_from_weights_file_ignores_artifact_metadata(tmp_path: Path) -> None:
    payload = {
        "model_type": "logistic_regression",
        "artifact_schema_version": 1,
        "feature_schema_version": 1,
        "feature_schema_hash": "not-validated-here",
        "card_catalog_hash": "not-validated-here",
        "feature_names": list(MODEL_FEATURE_NAMES),
        "coefficients": [0.0] * len(MODEL_FEATURE_NAMES),
        "intercept": 0.0,
    }
    path = tmp_path / "weights.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    model = LinearRunStateValueModel.from_weights_file(path, extractor=_extractor())

    assert model.predict_win_probability({}) == pytest.approx(0.5)


def test_unknown_feature_name_is_rejected_at_prediction_time() -> None:
    model = LinearRunStateValueModel(
        feature_names=["not_a_feature"],
        coefficients=[1.0],
        intercept=0.0,
        extractor=_extractor(),
    )

    with pytest.raises(ValueError, match="unknown feature name"):
        model.predict_win_probability({})
