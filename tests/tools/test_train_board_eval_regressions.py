from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import train_board_eval

from sts2_training.board_eval.card_features import CardFeatureExtractor
from sts2_training.board_eval.linear_value_function import LinearRunStateValueModel
from sts2_training.board_eval.training_data import MODEL_FEATURE_NAMES


class _Array(list[float]):
    def tolist(self) -> list[float]:
        return list(self)


class _FakePipeline:
    def __init__(self, coefficients: list[float], *, intercept: float = 0.0) -> None:
        size = len(coefficients)
        self.named_steps = {
            "standardscaler": SimpleNamespace(
                mean_=_Array([0.0] * size),
                scale_=_Array([1.0] * size),
            ),
            "logisticregression": SimpleNamespace(
                coef_=[_Array(coefficients)],
                intercept_=[intercept],
            ),
        }


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


def _decision_event(deck_ids: list[str], *, state_kind: str = "Shop") -> dict:
    return {
        "event": "selection",
        "received": {
            "masked_emulator_dto": {
                "hp": 80,
                "maxHp": 80,
                "gold": 50,
                "actFloor": 5,
                "totalFloor": 51,
                "currentRoomType": state_kind,
                "deck": [{"id": card_id} for card_id in deck_ids],
            }
        },
        "request": {"decision_point_id": "d0", "operation": "commit_action"},
        "result": {"status": "completed"},
    }


def _terminal_event(outcome: str) -> dict:
    return {
        "event": "self_play_run_result",
        "outcome": outcome,
        "final_dto": {"outcome": outcome},
    }


def _write_run(
    path: Path,
    outcome: str,
    *,
    deck_ids: list[str] | None = None,
    state_kind: str = "Shop",
) -> None:
    events = [
        _decision_event(deck_ids or ["STRIKE"], state_kind=state_kind),
        _terminal_event(outcome),
    ]
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


def test_small_biased_split_single_class_evaluation_does_not_raise(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")

    paths: list[Path] = []
    for index in range(5):
        path = tmp_path / f"win-{index}.jsonl"
        _write_run(path, "victory")
        paths.append(path)
    lose = tmp_path / "lose-0.jsonl"
    _write_run(lose, "defeat")
    paths.append(lose)

    split = train_board_eval.split_logs_by_run(
        paths,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=11,
    )
    extractor = _extractor()
    train_examples = train_board_eval._examples_for(
        split.train,
        extractor,
        on_unknown_card="raise",
    )
    pipeline = train_board_eval.fit_model(
        train_examples,
        inverse_regularization=1.0,
        seed=11,
    )

    assert {example.label for example in train_examples} == {0, 1}
    for split_paths in (split.val, split.test):
        examples = train_board_eval._examples_for(
            split_paths,
            extractor,
            on_unknown_card="raise",
        )
        assert {example.label for example in examples} == {1}
        with caplog.at_level(logging.INFO):
            metrics = train_board_eval.evaluate_model(pipeline, examples)
        assert metrics["log_loss"] is None
        assert metrics["roc_auc"] is None

    assert "only one class" in caplog.text


def test_unknown_card_skip_features_match_training_and_inference_order(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    _write_run(path, "victory", deck_ids=["STRIKE", "UNKNOWN"])
    extractor = _extractor()
    example = train_board_eval._examples_for(
        [path],
        extractor,
        on_unknown_card="skip",
    )[0]

    model = LinearRunStateValueModel(
        feature_names=MODEL_FEATURE_NAMES,
        coefficients=[0.0] * len(MODEL_FEATURE_NAMES),
        intercept=0.0,
        extractor=extractor,
        on_unknown_card="skip",
    )
    dto = _decision_event(["STRIKE", "UNKNOWN"])["received"]["masked_emulator_dto"]

    training_vector = example.to_model_vector()
    inference_vector = model._vector_from_dto(dto)

    assert inference_vector == pytest.approx(training_vector)
    unknown_index = MODEL_FEATURE_NAMES.index("deck_unknown_card_count")
    ratio_index = MODEL_FEATURE_NAMES.index("deck_known_card_ratio")
    assert training_vector[unknown_index] == pytest.approx(1.0)
    assert training_vector[ratio_index] == pytest.approx(0.5)


def test_main_passes_same_state_kind_filter_to_all_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_path = log_dir / "run.jsonl"
    _write_run(log_path, "victory")
    card_csv = tmp_path / "cards.csv"
    card_csv.write_text("card_id\nSTRIKE\n", encoding="utf-8")
    output = tmp_path / "weights.json"

    split = train_board_eval.RunSplit(
        train=[log_path],
        val=[log_path],
        test=[log_path],
    )
    monkeypatch.setattr(train_board_eval, "split_logs_by_run", lambda *args, **kwargs: split)

    seen_state_kinds: list[tuple[str, ...]] = []

    def fake_examples_for(
        log_paths: list[Path],
        extractor: CardFeatureExtractor,
        *,
        on_unknown_card: str,
        state_kinds: list[str] | None = None,
    ) -> list:
        del log_paths, extractor, on_unknown_card
        seen_state_kinds.append(tuple(state_kinds or ()))
        return []

    fake_pipeline = object()
    monkeypatch.setattr(train_board_eval, "_examples_for", fake_examples_for)
    monkeypatch.setattr(
        train_board_eval,
        "fit_model",
        lambda examples, *, inverse_regularization, seed: fake_pipeline,
    )
    monkeypatch.setattr(
        train_board_eval,
        "evaluate_model",
        lambda pipeline, examples: {"count": len(examples)},
    )
    monkeypatch.setattr(
        train_board_eval,
        "weights_payload",
        lambda pipeline, metrics, **kwargs: {
            "feature_names": [],
            "coefficients": [],
            "intercept": 0.0,
        },
    )

    exit_code = train_board_eval.main(
        [
            "--log-dir",
            str(log_dir),
            "--output",
            str(output),
            "--card-csv",
            str(card_csv),
            "--state-kind",
            "Shop",
            "--state-kind",
            "RestSite",
        ]
    )

    assert exit_code == 0
    assert seen_state_kinds == [("Shop", "RestSite")] * 3


def test_feature_schema_hash_changes_when_model_feature_names_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = tmp_path / "cards.csv"
    catalog.write_text("card_id\nSTRIKE\n", encoding="utf-8")
    original_names = tuple(train_board_eval.MODEL_FEATURE_NAMES)

    original_payload = train_board_eval.weights_payload(
        _FakePipeline([0.0] * len(original_names)),
        {"train": {"count": 0}},
        card_catalog_path=catalog,
    )

    changed_names = original_names + ("synthetic_contract_feature",)
    monkeypatch.setattr(train_board_eval, "MODEL_FEATURE_NAMES", changed_names)
    changed_payload = train_board_eval.weights_payload(
        _FakePipeline([0.0] * len(changed_names)),
        {"train": {"count": 0}},
        card_catalog_path=catalog,
    )

    assert original_payload["feature_schema_hash"] != changed_payload["feature_schema_hash"]
    assert changed_payload["feature_names"] == list(changed_names)
