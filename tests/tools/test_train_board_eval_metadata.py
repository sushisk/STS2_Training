from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import train_board_eval

from sts2_training.board_eval.training_data import MODEL_FEATURE_NAMES


class _Array(list[float]):
    def tolist(self) -> list[float]:
        return list(self)


class _FakePipeline:
    def __init__(self) -> None:
        size = len(MODEL_FEATURE_NAMES)
        self.named_steps = {
            "standardscaler": SimpleNamespace(
                mean_=_Array([0.0] * size),
                scale_=_Array([1.0] * size),
            ),
            "logisticregression": SimpleNamespace(
                coef_=[_Array([0.0] * size)],
                intercept_=[0.0],
            ),
        }


def test_weights_payload_records_canonical_state_kind_training_scope(tmp_path: Path) -> None:
    catalog = tmp_path / "cards.csv"
    catalog.write_text("card_id\nSTRIKE\n", encoding="utf-8")

    payload = train_board_eval.weights_payload(
        _FakePipeline(),
        {"train": {"count": 0}},
        card_catalog_path=catalog,
        training_state_kinds=["Shop", "RestSite", "Shop"],
    )

    assert payload["training_state_kinds"] == ["RestSite", "Shop"]


def test_weights_payload_marks_unfiltered_state_kind_scope_as_none(tmp_path: Path) -> None:
    catalog = tmp_path / "cards.csv"
    catalog.write_text("card_id\nSTRIKE\n", encoding="utf-8")

    payload = train_board_eval.weights_payload(
        _FakePipeline(),
        {"train": {"count": 0}},
        card_catalog_path=catalog,
    )

    assert payload["training_state_kinds"] is None
