from types import SimpleNamespace
import pytest
from sts2_training.runner.stable_pruner_rl_reward import paired_pruner_reward

def _arm(outcome, hp, potions, *, nodes=10, beam_ms=10.0):
    return SimpleNamespace(outcome=outcome, terminal_hp=hp, terminal_max_hp=100.0,
        terminal_potion_count=potions, initial_potion_count=2,
        nodes_expanded=nodes, beam_total_ms=beam_ms)

def _pair(baseline, learned):
    return SimpleNamespace(baseline=baseline, learned=learned)

def test_same_victory_rewards_post_combat_resources():
    pair = _pair(_arm("victory", 10.0, 0), _arm("victory", 80.0, 2))
    reward = paired_pruner_reward(pair)
    assert reward is not None
    assert reward.outcome_delta == 0.0
    assert reward.baseline_resource_quality == pytest.approx(0.08)
    assert reward.learned_resource_quality == pytest.approx(0.84)
    assert reward.resource_quality_delta == pytest.approx(0.76)
    assert reward.total == pytest.approx(0.19)

def test_terminal_outcome_remains_primary_signal():
    pair = _pair(_arm("victory", 1.0, 0), _arm("defeat", 100.0, 2))
    reward = paired_pruner_reward(pair)
    assert reward is not None
    assert reward.total < 0.0

def test_search_cost_penalties_remain_additive():
    pair = _pair(_arm("defeat", 50.0, 1, nodes=10), _arm("victory", 50.0, 1, nodes=12))
    reward = paired_pruner_reward(pair, node_cost_weight=0.1)
    assert reward is not None
    assert reward.total == pytest.approx(0.8)

def test_unknown_outcome_is_not_learnable():
    pair = _pair(_arm("victory", 50.0, 1), _arm("unknown", 50.0, 1))
    assert paired_pruner_reward(pair) is None
