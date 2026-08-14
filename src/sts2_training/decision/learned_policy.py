"""Dependency-free linear Combat ``action_score`` policy inference."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sts2_training.decision.action_score_features import (
    ACTION_SCORE_FEATURE_NAMES,
    ACTION_SCORE_FEATURE_SCHEMA_VERSION,
    combat_action_score_feature_matrix,
    combat_action_score_features,
)
from sts2_training.decision.oracle_log import (
    ORACLE_VALUE_MASK_VERSION,
    require_oracle_value_dto_version,
)
from sts2_training.decision.policy import ActionCandidate, PolicyModel

LEARNED_ACTION_SCORE_ARTIFACT_SCHEMA_VERSION = 1
ACTION_SCORE_MODEL_TYPE = "pairwise_logistic_action_score"


class LinearActionScorePolicy(PolicyModel):
    """Linear, pre-simulation candidate ranker distilled from Oracle root-action Q order."""

    def __init__(
        self,
        *,
        coefficients: Sequence[float],
        intercept: float,
        mean: Sequence[float],
        scale: Sequence[float],
        required_dto_version: str,
        artifact_sha256: str | None = None,
    ) -> None:
        expected = len(ACTION_SCORE_FEATURE_NAMES)
        if not (len(coefficients) == len(mean) == len(scale) == expected):
            raise ValueError("learned action_score vector lengths must match feature schema")
        if not isinstance(required_dto_version, str) or not required_dto_version:
            raise ValueError("learned action_score required_dto_version must be non-empty")
        self._coefficients = tuple(_finite(value, "coefficient") for value in coefficients)
        self._intercept = _finite(intercept, "intercept")
        self._mean = tuple(_finite(value, "mean") for value in mean)
        self._scale = tuple(_finite(value, "scale") for value in scale)
        if any(value <= 0.0 for value in self._scale):
            raise ValueError("learned action_score scale entries must be positive")
        self._required_dto_version = required_dto_version
        self._artifact_sha256 = artifact_sha256

    @classmethod
    def from_weights_file(cls, path: str | Path) -> "LinearActionScorePolicy":
        path = Path(path)
        raw = path.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("learned action_score artifact must be a JSON object")
        if payload.get("model_type") != ACTION_SCORE_MODEL_TYPE:
            raise ValueError(f"unsupported learned action_score model_type: {payload.get('model_type')!r}")
        if payload.get("artifact_schema_version") != LEARNED_ACTION_SCORE_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported learned action_score artifact_schema_version: "
                f"{payload.get('artifact_schema_version')!r}"
            )
        if payload.get("feature_schema_version") != ACTION_SCORE_FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "learned action_score feature schema mismatch: "
                f"{payload.get('feature_schema_version')!r}"
            )
        if payload.get("required_mask_version") != ORACLE_VALUE_MASK_VERSION:
            raise ValueError(
                "learned action_score mask contract mismatch: "
                f"{payload.get('required_mask_version')!r}"
            )
        required_dto_version = payload.get("required_dto_version")
        if not isinstance(required_dto_version, str) or not required_dto_version:
            raise ValueError("learned action_score artifact requires exact dto_version metadata")
        if tuple(payload.get("feature_names") or ()) != ACTION_SCORE_FEATURE_NAMES:
            raise ValueError("learned action_score feature_names do not match runtime schema")
        return cls(
            coefficients=_sequence(payload.get("coefficients"), "coefficients"),
            intercept=payload.get("intercept"),
            mean=_sequence(payload.get("mean"), "mean"),
            scale=_sequence(payload.get("scale"), "scale"),
            required_dto_version=required_dto_version,
            artifact_sha256=hashlib.sha256(raw).hexdigest(),
        )

    def score_action(
        self,
        action: Mapping[str, Any],
        masked_emulator_dto: Mapping[str, Any],
    ) -> float:
        require_oracle_value_dto_version(
            masked_emulator_dto,
            expected=self._required_dto_version,
            context="learned action_score runtime input",
        )
        features = combat_action_score_features(masked_emulator_dto, action)
        return self._score_features(features)

    def propose(
        self,
        legal_actions: Sequence[Mapping[str, Any]],
        masked_emulator_dto: Mapping[str, Any],
        *,
        top_k: int,
    ) -> list[ActionCandidate]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        # Validate the DTO generation even when there are no usable actions. A learned
        # artifact must never silently look compatible merely because ranking was empty.
        require_oracle_value_dto_version(
            masked_emulator_dto,
            expected=self._required_dto_version,
            context="learned action_score runtime input",
        )

        available: list[tuple[int, Mapping[str, Any], str]] = []
        for index, action in enumerate(legal_actions):
            if action.get("is_available") is False:
                continue
            action_id = action.get("action_id")
            action_type = action.get("action_type")
            if not isinstance(action_id, str) or not action_id or not isinstance(action_type, str):
                continue
            available.append((index, action, action_id))
        if not available:
            return []

        feature_rows = combat_action_score_feature_matrix(
            masked_emulator_dto, [action for _, action, _ in available]
        )
        scored = [
            (self._score_features(features), index, action_id)
            for (index, _action, action_id), features in zip(available, feature_rows, strict=True)
        ]
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [
            ActionCandidate(action_id=action_id, action_score=score)
            for score, _, action_id in scored[:top_k]
        ]

    def _score_features(self, features: Sequence[float]) -> float:
        score = self._intercept
        for index, value in enumerate(features):
            normalized = (value - self._mean[index]) / self._scale[index]
            score += self._coefficients[index] * normalized
        return float(score)

    def oracle_provenance(self) -> Mapping[str, Any]:
        return {
            "model_type": ACTION_SCORE_MODEL_TYPE,
            "artifact_schema_version": LEARNED_ACTION_SCORE_ARTIFACT_SCHEMA_VERSION,
            "feature_schema_version": ACTION_SCORE_FEATURE_SCHEMA_VERSION,
            "required_mask_version": ORACLE_VALUE_MASK_VERSION,
            "required_dto_version": self._required_dto_version,
            "artifact_sha256": self._artifact_sha256,
        }


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"learned action_score {field} must be a sequence")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"learned action_score {field} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"learned action_score {field} must be finite")
    return number


__all__ = [
    "ACTION_SCORE_MODEL_TYPE",
    "LEARNED_ACTION_SCORE_ARTIFACT_SCHEMA_VERSION",
    "LinearActionScorePolicy",
]
