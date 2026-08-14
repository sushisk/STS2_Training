from typing import Any

from sts2_training.runner.stable_pruner_rl_reward import (
    paired_pruner_reward as resource_reward,
)


def paired_pruner_reward(
    pair: Any,
    *,
    node_cost_weight: float = 0.0,
    beam_ms_cost_weight: float = 0.0,
) -> Any:
    return resource_reward(
        pair,
        node_cost_weight=node_cost_weight,
        beam_ms_cost_weight=beam_ms_cost_weight,
    )
