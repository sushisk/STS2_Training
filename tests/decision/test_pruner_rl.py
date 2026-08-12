from __future__ import annotations

import math

import pytest

from sts2_training.decision.learned_pruner import LinearStableFrontierPruner
from sts2_training.decision.pruner_features import (
    PRUNER_FEATURE_NAMES,
    PRUNER_FEATURE_SCHEMA_VERSION,
)
from sts2_training.decision.pruner_rl import (
    InMemoryPrunerRLStepCollector,
    PlackettLuceLinearStableFrontierPruner,
    plackett_luce_log_probability,
    plackett_luce_logprob_gradient,
)
from sts2_training.decision.stable_pruner import (
    STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
    StablePruneContext,
    StablePruneNodeView,
)


def _node(value: float, root_action_id: str) -> StablePruneNodeView:
    return StablePruneNodeView(
        value=value,
        root_action_id=root_action_id,
        depth=1,
        combat_depth=1,
        continuation_steps=0,
        terminal=False,
        action_type="card",
        policy_rank=0,
        policy_score=None,
        post_coverage_rank=0,
        candidate_source="policy",
    )


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
    frontier = [_node(4.0, "a"), _node(3.0, "b"), _node(2.0, "c"), _node(1.0, "d")]
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
    assert all(isinstance(index, int) for index in first_selected)
    assert len(first_collector.steps) == 1
    assert first_collector.steps == second_collector.steps
    step = first_collector.steps[0]
    assert step.stable_prune_node_view_schema_version == STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION
    assert step.feature_schema_version == PRUNER_FEATURE_SCHEMA_VERSION
    assert len(step.sampled_indices) == 2
    assert len(set(step.sampled_indices)) == 2
    assert set(step.returned_indices) == set(step.sampled_indices)
    returned_values = [frontier[index].value for index in step.returned_indices]
    assert returned_values == sorted(returned_values, reverse=True)
    assert tuple(first_selected) == step.returned_indices
    assert math.isfinite(step.selection_log_probability)


def test_no_rl_step_when_every_frontier_node_survives() -> None:
    collector = InMemoryPrunerRLStepCollector()
    pruner = PlackettLuceLinearStableFrontierPruner(
        _base_pruner(), seed=3, collector=collector
    )
    frontier = [_node(2.0, "a"), _node(1.0, "b")]

    selected = pruner.select(frontier, k=2, context=_context(beam_width=2))

    assert selected == [0, 1]
    assert collector.steps == []


def test_no_choice_still_returns_supervised_score_order() -> None:
    pruner = PlackettLuceLinearStableFrontierPruner(_base_pruner(), seed=3)
    frontier = [_node(1.0, "a"), _node(3.0, "b"), _node(2.0, "c")]

    assert pruner.select(frontier, k=3, context=_context(beam_width=3)) == [1, 2, 0]


def test_plackett_luce_log_probability_and_gradient_for_uniform_two_way_choice() -> None:
    scores = (0.0, 0.0)
    features = ((1.0,), (0.0,))

    log_probability = plackett_luce_log_probability(scores, (0,), temperature=1.0)
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
        PlackettLuceLinearStableFrontierPruner(_base_pruner(), temperature=temperature)
