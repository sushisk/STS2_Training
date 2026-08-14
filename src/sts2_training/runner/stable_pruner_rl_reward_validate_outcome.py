from collections.abc import Mapping
from typing import Any
from sts2_training.runner.stable_pruner_rl_reward_validate_fields import close, finite, integer, outcome

def validate_outcome_cost(record: Mapping[str, Any], reward: Mapping[str, Any], source: str, resource_weight: float, resource_delta: float) -> float:
    pair = record.get("paired_result")
    if not isinstance(pair, Mapping): raise ValueError(f"{source}: missing paired_result")
    bo, lo = outcome(pair.get("baseline_outcome")), outcome(pair.get("learned_outcome"))
    if bo is None or lo is None: raise ValueError(f"{source}: paired_result outcomes must be resolved")
    od = finite(reward.get("outcome_delta")); close(od, lo - bo, source, "reward.outcome_delta")
    nd = integer(reward.get("nodes_expanded_delta"))
    if nd != integer(pair.get("learned_nodes_expanded")) - integer(pair.get("baseline_nodes_expanded")):
        raise ValueError(f"{source}: reward.nodes_expanded_delta does not match paired_result")
    bd = finite(reward.get("beam_ms_delta"))
    close(bd, finite(pair.get("learned_beam_total_ms")) - finite(pair.get("baseline_beam_total_ms")), source, "reward.beam_ms_delta")
    nw, bw = finite(reward.get("node_cost_weight")), finite(reward.get("beam_ms_cost_weight"))
    if nw < 0 or bw < 0: raise ValueError(f"{source}: reward cost weights must be non-negative")
    total = finite(reward.get("total"))
    close(total, od + resource_weight * resource_delta - nw * nd - bw * bd, source, "reward.total")
    return total
