from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from sts2_training.decision.learned_pruner import LinearStableFrontierPruner
from sts2_training.decision.pruner_features import PRUNER_FEATURE_NAMES
from sts2_training.decision.pruner_rl import (
    InMemoryPrunerRLStepCollector,
    PlackettLuceLinearStableFrontierPruner,
    plackett_luce_log_probability,
    plackett_luce_logprob_gradient,
)
from sts2_training.decision.stable_pruner import StablePruneContext


@dataclass(frozen=True)
class _Node:
    value: float
    root_action_id: str | None
    depth: int = 1
    combat_depth: int = 1
    continuation_steps: int = 0
    terminal: bool = False
    action_type: str | None = "card"
    policy_rank: int | None = 0
    policy_score: float | None = None
    post_coverage_rank: int | None = 0
    candidate_source: str | None = "policy"


def _context(beam_width: int = 2) -> StablePruneContext:
    return StablePruneContext(
        search_id="search",
        prune_step_id="search:prune:0",
        phase="stable_frontier",
        beam_width=beam_width,
        max_depth=4,
        depths_completed=1,
        remaining_time_ms=100.0,
    )


def _base_pruner() -> LinearStableFrontierPruner:
    coefficients = [0.0] * len(PRUNER_FEATURE_NAMES)
    coefficients[PRUNER_FEATURE_NAMES.index("node_value")] = 1.0
    return LinearStableFrontierPruner(
        feature_names=PRUNER_FEATURE_NAMES,
        coefficients=coefficients,
    )


def test_plackett_luce_pruner_is_seed_reproducible_and_records_behavior_step() -> None:
    frontier = [
        _Node(4.0, "a"),
        _Node(3.0, "b"),
        _Node(2.0, "c"),
        _Node(1.0, "d"),
    ]
    first_collector = InMemoryPrunerRLStepCollector()
    second_collector = InMemoryPrunerRLStepCollector()
    first = PlackettLuceLinearStableFrontierPruner(
        _base_pruner(), temperature=1.5, seed=17, collector=first_collector
    )
    second = PlackettLuceLinearStableFrontierPruner(
        _base_pruner(), temperature=1.5, seed=17, collector=second_collector
    )

    first_selected = first.select(frontier, k=2, context=_context())
    second_selected = second.select(frontier, k=2, context=_context())

    assert first_selected == second_selected
    assert len(first_collector.steps) == 1
    assert first_collector.steps == second_collector.steps
    step = first_collector.steps[0]
    assert len(step.sampled_indices) == 2
    assert len(set(step.sampled_indices)) == 2
    assert set(step.returned_indices) == set(step.sampled_indices)
    returned_values = [frontier[index].value for index in step.returned_indices]
    assert returned_values == sorted(returned_values, reverse=True)
    assert math.isfinite(step.selection_log_probability)


def test_no_rl_step_when_every_frontier_node_survives() -> None:
    collector = InMemoryPrunerRLStepCollector()
    pruner = PlackettLuceLinearStableFrontierPruner(
        _base_pruner(), seed=3, collector=collector
    )
    frontier = [_Node(2.0, "a"), _Node(1.0, "b")]

    selected = pruner.select(frontier, k=2, context=_context(beam_width=2))

    assert [node.value for node in selected] == [2.0, 1.0]
    assert collector.steps == []


def test_plackett_luce_log_probability_and_gradient_for_uniform_two_way_choice() -> None:
    scores = (0.0, 0.0)
    features = ((1.0,), (0.0,))

    log_probability = plackett_luce_log_probability(
        scores, (0,), temperature=1.0
    )
    gradient = plackett_luce_logprob_gradient(
        scores,
        features,
        (0,),
        temperature=1.0,
        scale=(1.0,),
    )

    assert log_probability == pytest.approx(-math.log(2.0))
    assert gradient == pytest.approx((0.5,))


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf"), float("nan")])
def test_plackett_luce_pruner_rejects_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        PlackettLuceLinearStableFrontierPruner(
            _base_pruner(), temperature=temperature
        )
