"""Dependency-free linear Combat ValueModel inference."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sts2_training.decision.oracle_log import (
    ORACLE_VALUE_MASK_VERSION,
    require_oracle_value_dto_version,
)
from sts2_training.decision.value import ValueModel
from sts2_training.decision.value_features import (
    VALUE_FEATURE_NAMES,
    VALUE_FEATURE_SCHEMA_VERSION,
    combat_value_features,
)

# v3 pins the exact public Emulator dto_version in addition to mask/feature schemas.
LEARNED_VALUE_ARTIFACT_SCHEMA_VERSION = 3


class LinearValueModel(ValueModel):
    """Ridge-trained linear value model over one exact public Combat DTO generation."""

    def __init__(
        self,
        *,
        coefficients: Sequence[float],
        intercept: float,
        mean: Sequence[float],
        scale: Sequence[float],
        required_dto_version: str,
        terminal_values: Mapping[str, float] | None = None,
        artifact_sha256: str | None = None,
    ) -> None:
        expected = len(VALUE_FEATURE_NAMES)
        if not (len(coefficients) == len(mean) == len(scale) == expected):
            raise ValueError("learned ValueModel vector lengths must match feature schema")
        if not isinstance(required_dto_version, str) or not required_dto_version:
            raise ValueError("learned ValueModel required_dto_version must be non-empty")
        self._coefficients = tuple(_finite(value, "coefficient") for value in coefficients)
        self._intercept = _finite(intercept, "intercept")
        self._mean = tuple(_finite(value, "mean") for value in mean)
        self._scale = tuple(_finite(value, "scale") for value in scale)
        if any(value <= 0.0 for value in self._scale):
            raise ValueError("learned ValueModel scale entries must be positive")
        self._required_dto_version = required_dto_version
        self._terminal_values = {
            key: _finite(value, f"terminal_values[{key!r}]")
            for key, value in (terminal_values or {}).items()
            if key in {"victory", "defeat"}
        }
        self._artifact_sha256 = artifact_sha256

    @classmethod
    def from_weights_file(cls, path: str | Path) -> "LinearValueModel":
        path = Path(path)
        raw = path.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("learned ValueModel artifact must be a JSON object")
        if payload.get("artifact_schema_version") != LEARNED_VALUE_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported learned ValueModel artifact_schema_version: "
                f"{payload.get('artifact_schema_version')!r}"
            )
        if payload.get("feature_schema_version") != VALUE_FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "learned ValueModel feature schema mismatch: "
                f"{payload.get('feature_schema_version')!r}"
            )
        if payload.get("required_mask_version") != ORACLE_VALUE_MASK_VERSION:
            raise ValueError(
                "learned ValueModel mask contract mismatch: "
                f"{payload.get('required_mask_version')!r}"
            )
        required_dto_version = _artifact_required_dto_version(payload)
        if tuple(payload.get("feature_names") or ()) != VALUE_FEATURE_NAMES:
            raise ValueError("learned ValueModel feature_names do not match runtime schema")
        terminal_values = payload.get("terminal_values")
        if terminal_values is not None and not isinstance(terminal_values, Mapping):
            raise ValueError("learned ValueModel terminal_values must be an object")
        return cls(
            coefficients=_sequence(payload.get("coefficients"), "coefficients"),
            intercept=payload.get("intercept"),
            mean=_sequence(payload.get("mean"), "mean"),
            scale=_sequence(payload.get("scale"), "scale"),
            required_dto_version=required_dto_version,
            terminal_values=terminal_values,
            artifact_sha256=hashlib.sha256(raw).hexdigest(),
        )

    def evaluate(self, masked_emulator_dto: Mapping[str, Any]) -> float:
        exact = self.exact_terminal_utility(masked_emulator_dto)
        if exact is not None:
            return exact
        features = combat_value_features(masked_emulator_dto)
        score = self._intercept
        for index, value in enumerate(features):
            normalized = (value - self._mean[index]) / self._scale[index]
            score += self._coefficients[index] * normalized
        return float(score)

    def evaluate_batch(self, dtos: Sequence[Mapping[str, Any]]) -> list[float]:
        return [self.evaluate(dto) for dto in dtos]

    def exact_terminal_utility(
        self, masked_emulator_dto: Mapping[str, Any]
    ) -> float | None:
        # Validate contract before terminal short-circuiting; otherwise an old/new Emulator
        # DTO could bypass feature extraction and still be accepted for terminal states.
        require_oracle_value_dto_version(
            masked_emulator_dto,
            expected=self._required_dto_version,
            context="learned ValueModel runtime input",
        )
        outcome = _terminal_outcome(masked_emulator_dto)
        return None if outcome is None else self._terminal_values.get(outcome)

    def oracle_provenance(self) -> Mapping[str, Any]:
        return {
            "model_type": "ridge_linear_combat_value",
            "artifact_schema_version": LEARNED_VALUE_ARTIFACT_SCHEMA_VERSION,
            "feature_schema_version": VALUE_FEATURE_SCHEMA_VERSION,
            "required_mask_version": ORACLE_VALUE_MASK_VERSION,
            "required_dto_version": self._required_dto_version,
            "artifact_sha256": self._artifact_sha256,
            "terminal_values": dict(self._terminal_values),
        }


def _artifact_required_dto_version(payload: Mapping[str, Any]) -> str:
    """Resolve the exact DTO generation without discarding metadata from older writers.

    New writers may promote ``required_dto_version`` to the artifact top level.  The
    current supervised trainer already preserves the same value in
    ``metrics.train.dto_version`` via dataset stats, so v3 accepts that canonical fallback
    rather than losing the generation pin merely because the field has not been promoted.
    """

    direct = payload.get("required_dto_version")
    if isinstance(direct, str) and direct:
        return direct
    metrics = payload.get("metrics")
    train = metrics.get("train") if isinstance(metrics, Mapping) else None
    derived = train.get("dto_version") if isinstance(train, Mapping) else None
    if isinstance(derived, str) and derived:
        return derived
    raise ValueError("learned ValueModel artifact requires exact dto_version metadata")


def _terminal_outcome(dto: Mapping[str, Any]) -> str | None:
    outcome = dto.get("outcome")
    if outcome in {"victory", "run_victory"}:
        return "victory"
    if outcome == "defeat":
        return "defeat"
    transition = dto.get("transition")
    if isinstance(transition, Mapping) and transition.get("kind") == "combat_completed":
        victory = transition.get("victory")
        if victory is True:
            return "victory"
        if victory is False:
            return "defeat"
    return None


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"learned ValueModel {field} must be a sequence")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"learned ValueModel {field} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"learned ValueModel {field} must be finite")
    return number
