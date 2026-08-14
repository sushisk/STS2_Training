from __future__ import annotations
import math
from collections.abc import Mapping
from typing import Any
from sts2_training.runner.combat_resource_reward import CombatResourceSnapshot, combat_resource_quality

def finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("reward numeric field must be finite")
    return float(value)

def integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int): raise ValueError("reward integer field must be integer")
    return value

def close(actual: float, expected: float, source: str, field: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{source}: {field} does not match recomputed trajectory value")

def outcome(value: Any) -> float | None:
    if not isinstance(value, str): return None
    value = value.strip().lower()
    return 1.0 if value in {"victory", "win"} else 0.0 if value in {"defeat", "loss"} else None

def quality(reward: Mapping[str, Any], prefix: str, source: str) -> float:
    snapshot = CombatResourceSnapshot(finite(reward.get(f"{prefix}_terminal_hp")), finite(reward.get(f"{prefix}_terminal_max_hp")), integer(reward.get(f"{prefix}_terminal_potion_count")), integer(reward.get(f"{prefix}_initial_potion_count")))
    score = combat_resource_quality(snapshot)
    close(finite(reward.get(f"{prefix}_resource_quality")), score, source, f"reward.{prefix}_resource_quality")
    return score
