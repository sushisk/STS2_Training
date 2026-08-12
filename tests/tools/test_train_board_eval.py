from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import train_board_eval

from sts2_training.board_eval.card_features import CardFeatureExtractor
from sts2_training.board_eval.linear_value_function import LinearRunStateValueModel
from sts2_training.board_eval.training_data import MODEL_FEATURE_NAMES


def _extractor() -> CardFeatureExtractor:
    return CardFeatureExtractor(
        [
            {
                "card_id": "STRIKE",
                "card_type": "Attack",
                "target_type": "AnyEnemy",
                "energy_cost": "1",
                "damage": "6",
            },
            {
                "card_id": "POMMEL",
                "card_type": "Attack",
                "target_type": "AnyEnemy",
                "energy_cost": "1",
                "damage": "9",
                "cards_drawn": "1",
            },
        ]
    )


def _decision_event(deck_ids: list[str]) -> dict:
    return {
        "event": "selection",
        "received": {
            "masked_emulator_dto": {
                "hp": 80,
                "maxHp": 80,
                "gold": 50,
                "actFloor": 5,
                "totalFloor": 51,
                "currentRoomType": "Shop",
                "deck": [{"id": card_id} for card_id in deck_ids],
            }
        },
        "request": {"decision_point_id": "d0", "operation": "commit_action"},
        "result": {"status": "completed"},
    }


def _terminal_event(outcome: str) -> dict:
    return {
        "event": "selection",
        "received": None,
        "request": {"decision_point_id": "last", "operation": "commit_action"},
        "result": {"status": "completed", "masked_emulator_dto": {"outcome": outcome}},
        "run_result": {"outcome": outcome},
    }


def _write_run(path: Path, outcome: str, deck_ids: list[str] | None = None) -> None:
    events = [_decision_event(deck_ids or ["STRIKE"]), _terminal_event(outcome)]
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


def test_split_logs_by_run_is_stratified_by_outcome(tmp_path: Path) -> None:
    paths: list[Path] = []
    for index in range(10):
        win = tmp_path / f"win-{index}.jsonl"
        lose = tmp_path / f"lose-{index}.jsonl"
        _write_run(win, "victory")
        _write_run(lose, "defeat")
        paths.extend([win, lose])

    split = train_board_eval.split_logs_by_run(
        paths,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=7,
    )

    assert len(split.train) == 12
    assert len(split.val) == 4
    assert len(split.test) == 4
    for group in (split.train, split.val, split.test):
        labels = {
            train_board_eval.label_from_events(list(train_board_eval.iter_log_events(path)))
            for path in group
        }
        assert labels == {0, 1}


def test_split_logs_by_run_is_deterministic(tmp_path: Path) -> None:
    paths = []
    for index in range(4):
        path = tmp_path / f"run-{index}.jsonl"
        _write_run(path, "victory" if index % 2 else "defeat")
        paths.append(path)

    first = train_board_eval.split_logs_by_run(paths, val_fraction=0.25, test_fraction=0.25, seed=3)
    second = train_board_eval.split_logs_by_run(paths, val_fraction=0.25, test_fraction=0.25, seed=3)

    assert first == second


def test_split_logs_by_run_rejects_invalid_fractions() -> None:
    with pytest.raises(ValueError):
        train_board_eval.split_logs_by_run([], val_fraction=0.5, test_fraction=0.5, seed=0)
    with pytest.raises(ValueError):
        train_board_eval.split_logs_by_run([], val_fraction=-0.1, test_fraction=0.0, seed=0)


def _fitted_pipeline(tmp_path: Path):
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    paths: list[Path] = []
    for index in range(6):
        weak = tmp_path / f"weak-{index}.jsonl"
        strong = tmp_path / f"strong-{index}.jsonl"
        _write_run(weak, "defeat", ["STRIKE"])
        _write_run(strong, "victory", ["STRIKE", "POMMEL", "POMMEL"])
        paths.extend([weak, strong])
    examples = train_board_eval._examples_for(paths, _extractor(), on_unknown_card="raise")
    pipeline = train_board_eval.fit_model(examples, inverse_regularization=1.0, seed=0)
    return pipeline, examples


def test_evaluate_model_marks_single_class_metrics_undefined(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    pipeline, _ = _fitted_pipeline(tmp_path)
    lose_path = tmp_path / "eval-lose.jsonl"
    _write_run(lose_path, "defeat", ["STRIKE"])
    lose_examples = train_board_eval._examples_for(
        [lose_path],
        _extractor(),
        on_unknown_card="raise",
    )

    with caplog.at_level("INFO"):
        metrics = train_board_eval.evaluate_model(pipeline, lose_examples)

    assert metrics["log_loss"] is None
    assert metrics["roc_auc"] is None
    assert "only one class" in caplog.text


def test_weights_payload_has_schema_and_catalog_metadata(tmp_path: Path) -> None:
    pipeline, examples = _fitted_pipeline(tmp_path)
    catalog = tmp_path / "cards.csv"
    catalog.write_text("card_id\nSTRIKE\n", encoding="utf-8")

    payload = train_board_eval.weights_payload(
        pipeline,
        {"train": {"count": len(examples)}},
        card_catalog_path=catalog,
    )

    assert payload["model_type"] == "logistic_regression"
    assert payload["artifact_schema_version"] == 1
    assert payload["feature_schema_version"] == 1
    assert payload["feature_names"] == list(MODEL_FEATURE_NAMES)
    assert payload["card_catalog_hash"] == hashlib.sha256(catalog.read_bytes()).hexdigest()
    assert payload["card_catalog_path"] == str(catalog)
    assert payload["created_at"].endswith("Z")
    assert isinstance(payload["feature_schema_hash"], str)
    assert len(payload["feature_schema_hash"]) == 64
    assert payload["training_commit"] is None or isinstance(payload["training_commit"], str)


def test_weights_round_trip_through_run_state_model(tmp_path: Path) -> None:
    pipeline, examples = _fitted_pipeline(tmp_path)
    catalog = tmp_path / "cards.csv"
    catalog.write_text("card_id\nSTRIKE\n", encoding="utf-8")
    payload = train_board_eval.weights_payload(
        pipeline,
        {"train": {"count": len(examples)}},
        card_catalog_path=catalog,
    )
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps(payload), encoding="utf-8")

    model = LinearRunStateValueModel.from_weights_file(weights, extractor=_extractor())
    weak = model.predict_win_probability({"deck": [{"id": "STRIKE"}]})
    strong = model.predict_win_probability(
        {"deck": [{"id": "STRIKE"}, {"id": "POMMEL"}, {"id": "POMMEL"}]}
    )

    assert strong > weak


def test_examples_for_can_filter_state_kind(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    _write_run(path, "victory")

    examples = train_board_eval._examples_for(
        [path],
        _extractor(),
        on_unknown_card="raise",
        state_kinds={"RestSite"},
    )

    assert examples == []
