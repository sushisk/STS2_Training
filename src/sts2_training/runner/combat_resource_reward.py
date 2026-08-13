"""Frozen post-combat resource evaluator for stable-pruner RL."""
from __future__ import annotations
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

COMBAT_RESOURCE_EVALUATOR_VERSION = 1
COMBAT_RESOURCE_HP_WEIGHT = 0.8
COMBAT_RESOURCE_POTION_WEIGHT = 0.2
COMBAT_RESOURCE_REWARD_WEIGHT = 0.25

@dataclass(frozen=True)
class CombatResourceSnapshot:
    hp: float
    max_hp: float
    potion_count: int
    initial_potion_count: int

def combat_resource_snapshot(dto: Mapping[str, Any], *, initial_potion_count: int) -> CombatResourceSnapshot:
    if isinstance(initial_potion_count, bool) or not isinstance(initial_potion_count, int) or initial_potion_count < 0:
        raise ValueError("initial_potion_count must be a non-negative integer")
    hp = _number(dto.get("hp"), "hp")
    max_hp = _number(dto.get("maxHp"), "maxHp")
    if hp < 0 or max_hp <= 0:
        raise ValueError("terminal HP fields are out of range")
    potions = dto.get("potions")
    if potions is None:
        potions = ()
    if not isinstance(potions, Sequence) or isinstance(potions, (str, bytes, bytearray)):
        raise ValueError("potions must be a sequence")
    for index, potion in enumerate(potions):
        if not isinstance(potion, Mapping):
            raise ValueError(f"potions[{index}] must be a mapping")
    return CombatResourceSnapshot(hp, max_hp, len(potions), initial_potion_count)

def combat_resource_quality(snapshot: CombatResourceSnapshot) -> float:
    if snapshot.hp < 0 or snapshot.max_hp <= 0 or snapshot.potion_count < 0 or snapshot.initial_potion_count < 0:
        raise ValueError("resource snapshot fields are out of range")
    hp_fraction = _clamp(snapshot.hp / snapshot.max_hp)
    potion_fraction = 1.0 if snapshot.initial_potion_count == 0 else _clamp(snapshot.potion_count / snapshot.initial_potion_count)
    return COMBAT_RESOURCE_HP_WEIGHT * hp_fraction + COMBAT_RESOURCE_POTION_WEIGHT * potion_fraction

def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result

def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
