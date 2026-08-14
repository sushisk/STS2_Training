from sts2_training.runner.stable_pruner_rl import RL_TRAJECTORY_SCHEMA_VERSION
from rl_v3_step_fixture import behavior, step
from rl_v3_reward_fixture import reward

def record(sha="expected"):
    return {"record_type": "stable_pruner_rl_episode", "record_schema_version": RL_TRAJECTORY_SCHEMA_VERSION,
        "behavior": behavior(sha), "reward": reward(),
        "paired_result": {"baseline_outcome": "defeat", "learned_outcome": "victory",
            "baseline_nodes_expanded": 10, "learned_nodes_expanded": 12,
            "baseline_beam_total_ms": 4.0, "learned_beam_total_ms": 7.0},
        "steps": [step()]}
