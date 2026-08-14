from types import SimpleNamespace
from sts2_training.decision.learned_pruner import LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION
from sts2_training.decision.pruner_features import PRUNER_FEATURE_SCHEMA_VERSION
from sts2_training.decision.pruner_rl import PrunerRLStep
from sts2_training.decision.stable_pruner import STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION
from sts2_training.runner.stable_pruner_rl import RL_TRAJECTORY_SCHEMA_VERSION, rl_episode_record
from sts2_training.runner.stable_pruner_rl_reward import paired_pruner_reward

def test_episode_record_carries_resource_reward_and_v3_schema():
    arm = lambda outcome: SimpleNamespace(outcome=outcome, terminal_hp=40.0, terminal_max_hp=50.0,
        terminal_potion_count=1, initial_potion_count=1, nodes_expanded=1, beam_total_ms=2.0)
    pair = SimpleNamespace(baseline=arm("defeat"), learned=arm("victory"), winner="learned",
        common_action_prefix=0, first_divergence_index=0)
    reward = paired_pruner_reward(pair)
    assert reward is not None
    report = SimpleNamespace(scenario_template_sha256="b" * 64, search_config={"beam_width": 1}, pairs=(pair,))
    pruner = SimpleNamespace(name="plackett_luce_linear_pruner", version="test", temperature=1.0,
        artifact_metadata={"artifact_sha256": "a" * 64,
            "artifact_schema_version": LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION,
            "stable_prune_node_view_schema_version": STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
            "feature_schema_version": PRUNER_FEATURE_SCHEMA_VERSION})
    step = PrunerRLStep(search_id="s", prune_step_id="s:0", phase="stable_frontier",
        beam_width=1, max_depth=2, depths_completed=1, remaining_time_ms=None,
        stable_prune_node_view_schema_version=STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
        feature_schema_version=PRUNER_FEATURE_SCHEMA_VERSION, temperature=1.0,
        sampler_seed=7, frontier_features=((1.0,), (0.0,)), behavior_scores=(1.0, 0.0),
        sampled_indices=(0,), returned_indices=(0,), selection_log_probability=-0.31326168751822286)
    record = rl_episode_record(seed=11, sampler_seed=7, report=report, reward=reward, pruner=pruner, steps=(step,))
    assert record["record_schema_version"] == RL_TRAJECTORY_SCHEMA_VERSION == 3
    assert record["reward"]["resource_evaluator_version"] == 1
    assert record["reward"]["baseline_terminal_hp"] == 40.0
