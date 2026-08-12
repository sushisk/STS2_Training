"""Standard-library inference for a trained linear Run-state value model."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from sts2_training.board_eval.card_features import CardFeatureExtractor
from sts2_training.board_eval.deck_summary import DECK_SUMMARY_FEATURE_NAMES
from sts2_training.board_eval.dto_adapter import UnknownCardPolicy, board_context_from_dto, deck_summary_from_dto
from sts2_training.board_eval.run_state_value import RunStateValueModel
from sts2_training.board_eval.training_data import context_model_features

__all__ = ["LinearRunStateValueModel"]


class LinearRunStateValueModel(RunStateValueModel):
    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        coefficients: Sequence[float],
        intercept: float,
        mean: Sequence[float] | None = None,
        scale: Sequence[float] | None = None,
        extractor: CardFeatureExtractor | None = None,
        on_unknown_card: UnknownCardPolicy = "skip",
    ) -> None:
        feature_names = tuple(feature_names)
        if len(feature_names) != len(coefficients):
            raise ValueError("feature_names and coefficients must be the same length")
        if mean is not None and len(mean) != len(feature_names):
            raise ValueError("mean must be the same length as feature_names when provided")
        if scale is not None and len(scale) != len(feature_names):
            raise ValueError("scale must be the same length as feature_names when provided")
        if scale is not None and any(value == 0 for value in scale):
            raise ValueError("scale entries must be non-zero")

        self._feature_names = feature_names
        self._coefficients = tuple(float(value) for value in coefficients)
        self._intercept = float(intercept)
        self._mean = tuple(float(value) for value in mean) if mean is not None else None
        self._scale = tuple(float(value) for value in scale) if scale is not None else None
        self._extractor = extractor if extractor is not None else CardFeatureExtractor.from_csv()
        self._on_unknown_card = on_unknown_card

    @classmethod
    def from_weights_file(
        cls,
        path: str | Path,
        *,
        extractor: CardFeatureExtractor | None = None,
        on_unknown_card: UnknownCardPolicy = "skip",
    ) -> LinearRunStateValueModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_names=payload["feature_names"],
            coefficients=payload["coefficients"],
            intercept=payload["intercept"],
            mean=payload.get("mean"),
            scale=payload.get("scale"),
            extractor=extractor,
            on_unknown_card=on_unknown_card,
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    def predict_win_probability(self, run_state: Mapping[str, object]) -> float:
        vector = self._vector_from_dto(run_state)
        logit = self._intercept
        for index, raw_value in enumerate(vector):
            value = raw_value
            if self._mean is not None and self._scale is not None:
                value = (value - self._mean[index]) / self._scale[index]
            logit += self._coefficients[index] * value
        return _sigmoid(logit)

    def _vector_from_dto(self, dto: Mapping[str, object]) -> tuple[float, ...]:
        by_name = context_model_features(board_context_from_dto(dto))
        summary = deck_summary_from_dto(dto, self._extractor, on_unknown_card=self._on_unknown_card)
        summary_vector = summary.to_vector()
        by_name.update(
            {
                f"deck_{name}": value
                for name, value in zip(DECK_SUMMARY_FEATURE_NAMES, summary_vector, strict=True)
            }
        )
        missing = [name for name in self._feature_names if name not in by_name]
        if missing:
            raise ValueError(f"weights file references unknown feature name(s): {missing}")
        return tuple(by_name[name] for name in self._feature_names)


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1.0 + exp_logit)
