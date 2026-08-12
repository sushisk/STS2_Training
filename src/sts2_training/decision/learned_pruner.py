"""Dependency-free runtime inference for a supervised stable-frontier ranker."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sts2_training.decision.pruner_features import (
    PRUNER_FEATURE_NAMES,
    PRUNER_FEATURE_SCHEMA_VERSION,
    stable_pruner_feature_matrix,
)
from sts2_training.decision.stable_pruner import (
    STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
    StableFrontierPruner,
    StablePruneContext,
    StablePruneNodeView,
)

LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION = 2


class LinearStableFrontierPruner(StableFrontierPruner):
    """Rank stable-node views with a learned linear score and return survivor indices."""

    name = "linear_learned_pruner"
    version = "1"

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        coefficients: Sequence[float],
        scale: Sequence[float] | None = None,
        artifact_metadata: dict[str, Any] | None = None,
    ) -> None:
        feature_names = tuple(str(name) for name in feature_names)
        if feature_names != PRUNER_FEATURE_NAMES:
            raise ValueError(
                "learned pruner artifact feature_names do not match the runtime feature schema"
            )
        if len(coefficients) != len(feature_names):
            raise ValueError("coefficients must match feature_names length")
        if scale is not None and len(scale) != len(feature_names):
            raise ValueError("scale must match feature_names length")

        coefficients_tuple = tuple(float(value) for value in coefficients)
        if not all(math.isfinite(value) for value in coefficients_tuple):
            raise ValueError("coefficients must be finite")
        scale_tuple = (
            tuple(float(value) for value in scale)
            if scale is not None
            else tuple(1.0 for _ in feature_names)
        )
        if not all(math.isfinite(value) and value > 0.0 for value in scale_tuple):
            raise ValueError("scale entries must be finite positive numbers")

        self._feature_names = feature_names
        self._coefficients = coefficients_tuple
        self._scale = scale_tuple
        self._artifact_metadata = dict(artifact_metadata or {})
        artifact_sha = self._artifact_metadata.get("artifact_sha256")
        if isinstance(artifact_sha, str) and artifact_sha:
            self.version = f"artifact:{artifact_sha[:12]}"

    @classmethod
    def from_weights_file(cls, path: str | Path) -> "LinearStableFrontierPruner":
        raw = Path(path).read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        if payload.get("model_type") != "pairwise_logistic_linear_pruner":
            raise ValueError("unsupported learned pruner model_type")
        if payload.get("artifact_schema_version") != LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported learned pruner artifact_schema_version")
        if (
            payload.get("stable_prune_node_view_schema_version")
            != STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION
        ):
            raise ValueError("learned pruner stable_prune_node_view_schema_version mismatch")
        if payload.get("feature_schema_version") != PRUNER_FEATURE_SCHEMA_VERSION:
            raise ValueError("learned pruner feature_schema_version mismatch")
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"feature_names", "coefficients", "scale"}
        }
        metadata["artifact_sha256"] = hashlib.sha256(raw).hexdigest()
        return cls(
            feature_names=payload["feature_names"],
            coefficients=payload["coefficients"],
            scale=payload.get("scale"),
            artifact_metadata=metadata,
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def artifact_metadata(self) -> dict[str, Any]:
        return dict(self._artifact_metadata)

    def score_features(self, features: Sequence[float]) -> float:
        """Score one already-featurized node, for offline evaluation/replay tooling."""
        return self._score_row(features)

    def score_batch(
        self,
        frontier: Sequence[StablePruneNodeView],
        *,
        context: StablePruneContext,
    ) -> list[float]:
        rows = stable_pruner_feature_matrix(frontier, context=context)
        return [self.score_features(row) for row in rows]

    def select(
        self,
        frontier: Sequence[StablePruneNodeView],
        *,
        k: int,
        context: StablePruneContext,
    ) -> list[int]:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        scores = self.score_batch(frontier, context=context)
        # Python's sort is stable, so score ties preserve authoritative frontier order.
        order = sorted(range(len(frontier)), key=lambda index: scores[index], reverse=True)
        return order[:k]

    def _score_row(self, row: Sequence[float]) -> float:
        if len(row) != len(self._coefficients):
            raise ValueError("runtime feature vector length does not match learned artifact")
        return sum(
            coefficient * (float(value) / scale)
            for coefficient, value, scale in zip(
                self._coefficients,
                row,
                self._scale,
                strict=True,
            )
        )
