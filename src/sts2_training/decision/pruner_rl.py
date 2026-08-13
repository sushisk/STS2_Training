"""On-policy exploration primitives for stable-frontier pruning.

The supervised linear pruner remains the scoring model. This module adds a narrow
Plackett-Luce exploration wrapper that samples survivor sets without changing policy
candidate generation, continuation handling, beam width, or stopping.

The wrapper consumes only the public ``StablePruneNodeView`` contract owned by PR #47
and returns ordered frontier indices, matching ``StableFrontierPruner`` exactly. Learned
ranking outputs are called ``frontier_score`` values to distinguish them from isolated
``state_score`` values and pre-simulation ``action_score`` values.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sts2_training.decision.learned_pruner import LinearStableFrontierPruner
from sts2_training.decision.pruner_features import (
    PRUNER_FEATURE_SCHEMA_VERSION,
    stable_pruner_feature_matrix,
)
from sts2_training.decision.stable_pruner import (
    STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
    StableFrontierPruner,
    StablePruneContext,
    StablePruneNodeView,
)


@dataclass(frozen=True)
class PrunerRLStep:
    """One stochastic stable-prune action under the behavior policy.

    ``behavior_scores`` is the persisted compatibility field name for the behavior
    policy's learned ``frontier_score`` values.
    """

    search_id: str
    prune_step_id: str
    phase: str
    beam_width: int
    max_depth: int
    depths_completed: int
    remaining_time_ms: float | None
    stable_prune_node_view_schema_version: int
    feature_schema_version: int
    temperature: float
    sampler_seed: int
    frontier_features: tuple[tuple[float, ...], ...]
    behavior_scores: tuple[float, ...]
    sampled_indices: tuple[int, ...]
    returned_indices: tuple[int, ...]
    selection_log_probability: float

    @property
    def behavior_frontier_scores(self) -> tuple[float, ...]:
        """Canonical name for the behavior policy's learned frontier scores."""

        return self.behavior_scores


class PrunerRLStepCollector(Protocol):
    def record(self, step: PrunerRLStep) -> None:
        ...


class InMemoryPrunerRLStepCollector:
    def __init__(self) -> None:
        self.steps: list[PrunerRLStep] = []

    def record(self, step: PrunerRLStep) -> None:
        self.steps.append(step)


class PlackettLuceLinearStableFrontierPruner(StableFrontierPruner):
    """Explore stable survivor sets around a trained linear pruner.

    The sampled Plackett-Luce order is the stochastic action used for policy-gradient
    accounting. Runtime returns the sampled *set* in deterministic base-frontier-score
    order so exploration does not additionally randomize parent expansion order.
    """

    name = "plackett_luce_linear_pruner"

    def __init__(
        self,
        base_pruner: LinearStableFrontierPruner,
        *,
        temperature: float = 1.0,
        seed: int = 0,
        collector: PrunerRLStepCollector | None = None,
    ) -> None:
        _validate_temperature(temperature)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self._base = base_pruner
        self._temperature = float(temperature)
        self._sampler_seed = seed
        self._rng = random.Random(seed)
        self._collector = collector
        self.version = f"{base_pruner.version}:plackett_luce:t={self._temperature:g}"

    @classmethod
    def from_weights_file(
        cls,
        path: str | Path,
        *,
        temperature: float = 1.0,
        seed: int = 0,
        collector: PrunerRLStepCollector | None = None,
    ) -> "PlackettLuceLinearStableFrontierPruner":
        return cls(
            LinearStableFrontierPruner.from_weights_file(path),
            temperature=temperature,
            seed=seed,
            collector=collector,
        )

    @property
    def artifact_metadata(self) -> dict[str, object]:
        return self._base.artifact_metadata

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def sampler_seed(self) -> int:
        return self._sampler_seed

    def frontier_scores(
        self,
        frontier: Sequence[StablePruneNodeView],
        *,
        context: StablePruneContext,
    ) -> list[float]:
        return self._base.frontier_scores(frontier, context=context)

    def score_batch(
        self,
        frontier: Sequence[StablePruneNodeView],
        *,
        context: StablePruneContext,
    ) -> list[float]:
        """Compatibility alias for :meth:`frontier_scores`."""

        return self.frontier_scores(frontier, context=context)

    def select(
        self,
        frontier: Sequence[StablePruneNodeView],
        *,
        k: int,
        context: StablePruneContext,
    ) -> list[int]:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if not frontier:
            return []

        rows = stable_pruner_feature_matrix(frontier, context=context)
        frontier_scores = tuple(
            self._base.frontier_score_features(row) for row in rows
        )
        if not all(math.isfinite(frontier_score) for frontier_score in frontier_scores):
            raise ValueError("stable pruner frontier scores must be finite")

        take = min(k, len(frontier))
        # Python's sort is stable, so frontier-score ties preserve authoritative order.
        base_order = tuple(
            sorted(
                range(len(frontier)),
                key=lambda index: frontier_scores[index],
                reverse=True,
            )
        )

        # No actual pruning choice exists when every node survives. Match the supervised
        # selector's ordered-index behavior and do not create a zero-information RL step.
        if take == len(frontier):
            return list(base_order)

        sampled_indices = _sample_plackett_luce_order(
            frontier_scores,
            take=take,
            temperature=self._temperature,
            rng=self._rng,
        )
        selected_set = set(sampled_indices)
        returned_indices = tuple(index for index in base_order if index in selected_set)
        log_probability = plackett_luce_log_probability(
            frontier_scores,
            sampled_indices,
            temperature=self._temperature,
        )

        if self._collector is not None:
            self._collector.record(
                PrunerRLStep(
                    search_id=context.search_id,
                    prune_step_id=context.prune_step_id,
                    phase=context.phase,
                    beam_width=context.beam_width,
                    max_depth=context.max_depth,
                    depths_completed=context.depths_completed,
                    remaining_time_ms=context.remaining_time_ms,
                    stable_prune_node_view_schema_version=(
                        STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION
                    ),
                    feature_schema_version=PRUNER_FEATURE_SCHEMA_VERSION,
                    temperature=self._temperature,
                    sampler_seed=self._sampler_seed,
                    frontier_features=tuple(rows),
                    behavior_scores=frontier_scores,
                    sampled_indices=sampled_indices,
                    returned_indices=returned_indices,
                    selection_log_probability=log_probability,
                )
            )
        return list(returned_indices)


def plackett_luce_log_probability(
    scores: Sequence[float],
    sampled_indices: Sequence[int],
    *,
    temperature: float,
) -> float:
    """Log probability of one ordered sample without replacement.

    ``scores`` is retained as the public parameter name for compatibility; callers in the
    stable-pruner domain should pass learned ``frontier_score`` values.
    """

    _validate_temperature(temperature)
    available = list(range(len(scores)))
    seen: set[int] = set()
    log_probability = 0.0
    for selected in sampled_indices:
        if (
            isinstance(selected, bool)
            or not isinstance(selected, int)
            or selected < 0
            or selected >= len(scores)
            or selected in seen
            or selected not in available
        ):
            raise ValueError("sampled_indices must be unique valid frontier indices")
        logits = [float(scores[index]) / float(temperature) for index in available]
        log_denominator = _logsumexp(logits)
        log_probability += float(scores[selected]) / float(temperature) - log_denominator
        available.remove(selected)
        seen.add(selected)
    return log_probability


def plackett_luce_logprob_gradient(
    scores: Sequence[float],
    feature_rows: Sequence[Sequence[float]],
    sampled_indices: Sequence[int],
    *,
    temperature: float,
    scale: Sequence[float],
) -> tuple[float, ...]:
    """Gradient of ordered log probability with respect to linear coefficients.

    ``scores`` is retained as the public parameter name for compatibility; it represents
    learned ``frontier_score`` values.
    """

    _validate_temperature(temperature)
    if len(scores) != len(feature_rows):
        raise ValueError("scores and feature_rows must have the same length")
    if not feature_rows:
        return tuple(0.0 for _ in scale)
    width = len(scale)
    if any(len(row) != width for row in feature_rows):
        raise ValueError("feature row width must match scale")
    normalized_scale = tuple(float(value) for value in scale)
    if not all(math.isfinite(value) and value > 0 for value in normalized_scale):
        raise ValueError("scale must contain finite positive numbers")

    phi = [
        tuple(float(value) / factor for value, factor in zip(row, normalized_scale, strict=True))
        for row in feature_rows
    ]
    gradient = [0.0] * width
    available = list(range(len(scores)))
    seen: set[int] = set()

    for selected in sampled_indices:
        if (
            isinstance(selected, bool)
            or not isinstance(selected, int)
            or selected < 0
            or selected >= len(scores)
            or selected in seen
            or selected not in available
        ):
            raise ValueError("sampled_indices must be unique valid frontier indices")
        probabilities = _softmax(
            [float(scores[index]) / float(temperature) for index in available]
        )
        inv_temperature = 1.0 / float(temperature)
        for feature_index in range(width):
            expected = sum(
                probability * phi[index][feature_index]
                for probability, index in zip(probabilities, available, strict=True)
            )
            gradient[feature_index] += (
                phi[selected][feature_index] - expected
            ) * inv_temperature
        available.remove(selected)
        seen.add(selected)

    return tuple(gradient)


def _sample_plackett_luce_order(
    frontier_scores: Sequence[float],
    *,
    take: int,
    temperature: float,
    rng: random.Random,
) -> tuple[int, ...]:
    _validate_temperature(temperature)
    if (
        isinstance(take, bool)
        or not isinstance(take, int)
        or take < 0
        or take > len(frontier_scores)
    ):
        raise ValueError("take must be in [0, len(frontier_scores)]")
    available = list(range(len(frontier_scores)))
    selected: list[int] = []
    for _ in range(take):
        probabilities = _softmax(
            [float(frontier_scores[index]) / float(temperature) for index in available]
        )
        threshold = rng.random()
        cumulative = 0.0
        chosen_position = len(available) - 1
        for position, probability in enumerate(probabilities):
            cumulative += probability
            if threshold < cumulative:
                chosen_position = position
                break
        selected.append(available.pop(chosen_position))
    return tuple(selected)


def _validate_temperature(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError("temperature must be a finite positive number")


def _softmax(logits: Sequence[float]) -> tuple[float, ...]:
    if not logits:
        return ()
    if not all(math.isfinite(float(value)) for value in logits):
        raise ValueError("Plackett-Luce logits must be finite")
    maximum = max(float(value) for value in logits)
    weights = [math.exp(float(value) - maximum) for value in logits]
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def _logsumexp(logits: Sequence[float]) -> float:
    if not logits:
        raise ValueError("logsumexp requires at least one value")
    maximum = max(float(value) for value in logits)
    return maximum + math.log(sum(math.exp(float(value) - maximum) for value in logits))
