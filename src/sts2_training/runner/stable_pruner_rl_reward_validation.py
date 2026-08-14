from collections.abc import Mapping
from typing import Any
from sts2_training.runner.stable_pruner_rl_reward_validate_resource import validate_resource
from sts2_training.runner.stable_pruner_rl_reward_validate_outcome import validate_outcome_cost

def validate_reward_record(record: Mapping[str, Any], reward: Mapping[str, Any], *, source: str) -> float:
    weight, delta = validate_resource(reward, source)
    return validate_outcome_cost(record, reward, source, weight, delta)
