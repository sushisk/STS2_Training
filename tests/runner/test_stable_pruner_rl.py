from __future__ import annotations

from types import SimpleNamespace

import pytest

from sts2_training.decision.learned_pruner import LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION
from sts2_training.decision.pruner_features import PRUNER_FEATURE_SCHEMA_VERSION
from sts2_training.decision.pruner_rl import PrunerRLStep
from sts2_training.decision.stable_pruner import STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION
from sts2_training.runner.stable_pruner_rl import (
    RL_TRAJECTORY_SCHEMA_VERSION,
    PairedPrunerReward,
    paired_pruner_reward,
    rl_episode_record,
)


def _pair(
    *,
    baseline_outcome: str = "victory",
    learned_outcome: str = "victory",
    baseline_nodes: int = 10,
    learned_nodes: int = 8,
    baseline_ms: float = 100.0,
    learned_ms: float = 90.0,
):
    return SimpleNamespace(
        baseline=SimpleNamespace(
            outcome=baseline_outcome,
            nodes_expanded=baseline_nodes,
            beam_total_ms=baseline_ms,
        ),
        learned=SimpleNamespace(
            outcome=learned_outcome,
            nodes_expanded=learned_nodes,
            beam_total_ms=learned_ms,
        ),
        winner="learned",
        common_action_prefix=0,
        first_divergence_index=0,
    )


def test_paired_reward_uses_terminal_delta_and_search_cost_delta() -> None:
    reward = paired_pruner_reward(
        _pair(
            baseline_outcome="defeat",
            learned_outcome="victory",
            baseline_nodes=10,
            learned_nodes=8,
            baseline_ms=100.0,
            learned_ms=90.0,
        ),
        node_cost_weight=0.1,
        beam_ms_cost_weight=0.01,
    )

    assert reward is not None
    assert reward.outcome_delta == 1.0
    assert reward.nodes_expanded_delta == -2
    assert reward.beam_ms_delta == -10.0
    assert reward.total == pytest.approx(1.3)


def test_paired_reward_skips_unknown_terminal_outcomes() -> None:
    assert paired_pruner_reward(_pair(learned_outcome="unknown")) is None


@pytest.mark.parametrize("weight", [-0.1, float("inf"), float("nan")])
def test_paired_reward_rejects_invalid_cost_weight(weight: float) -> None:
    with pytest.raises(ValueError):
        paired_pruner_reward(_pair(), node_cost_weight=weight)


def test_episode_record_carries_complete_behavior_schema_provenance() -> None:
    pruner = SimpleNamespace(
        name="plackett_luce_linear_pruner",
        version="artifact:abc:plackett_luce:t=1",
        temperature=1.0,
        artifact_metadata={
            "artifact_sha256": "a" * 64,
            "artifact_schema_version": LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION,
            "stable_prune_node_view_schema_version": STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
            "feature_schema_version": PRUNER_FEATURE_SCHEMA_VERSION,
        },
    )
    step = PrunerRLStep(
        search_id="search",
        prune_step_id="search:prune:0",
        phase="stable_frontier",
        beam_width=1,
        max_depth=2,
        depths_completed=1,
        remaining_time_ms=None,
        stable_prune_node_view_schema_version=STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
        feature_schema_version=PRUNER_FEATURE_SCHEMA_VERSION,
        temperature=1.0,
        sampler_seed=7,
        frontier_features=((1.0,), (0.0,)),
        behavior_scores=(1.0, 0.0),
        sampled_indices=(0,),
        returned_indices=(0,),
        selection_log_probability=-0.31326168751822286,
    )
    report = SimpleNamespace(
        scenario_template_sha256="b" * 64,
        search_config={"beam_width": 1},
        pairs=(_pair(),),
    )
    reward = PairedPrunerReward(1.0, 0, 0.0, 0.0, 0.0, 1.0)

    record = rl_episode_record(
        seed=11,
        sampler_seed=7,
        report=report,
        reward=reward,
        pruner=pruner,
        steps=(step,),
    )

    assert record["record_schema_version"] == RL_TRAJECTORY_SCHEMA_VERSION
    assert record["behavior"]["artifact_schema_version"] == LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION
    assert (
        record["behavior"]["stable_prune_node_view_schema_version"]
        == STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION
    )
    assert record["behavior"]["feature_schema_version"] == PRUNER_FEATURE_SCHEMA_VERSION
    assert (
        record["steps"][0]["stable_prune_node_view_schema_version"]
        == STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION
    )
