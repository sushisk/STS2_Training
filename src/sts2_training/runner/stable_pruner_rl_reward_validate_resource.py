from collections.abc import Mapping
from typing import Any
from sts2_training.runner.combat_resource_reward import COMBAT_RESOURCE_EVALUATOR_VERSION, COMBAT_RESOURCE_HP_WEIGHT, COMBAT_RESOURCE_POTION_WEIGHT, COMBAT_RESOURCE_REWARD_WEIGHT
from sts2_training.runner.stable_pruner_rl_reward_validate_fields import close, finite, integer, quality

def validate_resource(reward: Mapping[str, Any], source: str) -> tuple[float, float]:
    if integer(reward.get("resource_evaluator_version")) != COMBAT_RESOURCE_EVALUATOR_VERSION:
        raise ValueError(f"{source}: resource evaluator version mismatch")
    close(finite(reward.get("resource_hp_weight")), COMBAT_RESOURCE_HP_WEIGHT, source, "reward.resource_hp_weight")
    close(finite(reward.get("resource_potion_weight")), COMBAT_RESOURCE_POTION_WEIGHT, source, "reward.resource_potion_weight")
    weight = finite(reward.get("resource_reward_weight"))
    close(weight, COMBAT_RESOURCE_REWARD_WEIGHT, source, "reward.resource_reward_weight")
    delta = finite(reward.get("resource_quality_delta"))
    close(delta, quality(reward, "learned", source) - quality(reward, "baseline", source), source, "reward.resource_quality_delta")
    return weight, delta
