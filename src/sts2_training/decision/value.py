"""`ValueModel`: scores one Combat `masked_emulator_dto`.

`BeamSearchEngine` calls `evaluate_batch` once per beam depth, covering every
node scored at that depth in a single call - a real learned value net should
override `evaluate_batch` directly (one batched forward pass) to hit the
~1ms/batch latency budget this is designed around; the default `evaluate_batch`
here just loops over `evaluate`.

`HeuristicValueFunction` is the no-model default: hand-picked features (HP/
enemy-HP ratios, block, predicted incoming damage, buffs/debuffs) combined
through fixed weights, the same idea as the OLD Combat package's
`StateEvaluator` but rebuilt against the actual RL/Training wire schema
(`rl_training_dto_documentation.md` section 3, "そのまま公開する情報") instead
of the old in-process `engine_state` shape. It exists so `BeamSearchEngine`/
`CombatDecisionEngine` are runnable and testable before any trained value
checkpoint exists - swap in a real model by implementing `ValueModel`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

JsonObject = Mapping[str, Any]

DEFAULT_WEIGHTS: dict[str, float] = {
    "player_hp_ratio": 40.0,
    "player_block": 0.5,
    "enemy_hp_ratio": -30.0,
    "predicted_incoming_damage": -1.0,
    "enemies_alive": -2.0,
    "buff_debuff_score": 2.0,
    "victory_bonus": 100_000.0,
    "defeat_penalty": -100_000.0,
}


class ValueModel:
    """Scores one `masked_emulator_dto`; higher is better for Training.

    Implement `evaluate` for scalar inference, or override `evaluate_batch`
    directly for a batch-only learned model. Batch-only implementations do not
    need a dummy scalar method just to satisfy an abstract base class.
    """

    def evaluate(self, masked_emulator_dto: Mapping[str, Any]) -> float:
        raise NotImplementedError("ValueModel must override evaluate or evaluate_batch")

    def evaluate_batch(self, dtos: Sequence[Mapping[str, Any]]) -> list[float]:
        """Batched counterpart of `evaluate`, one entry per dto, in order.

        Override this directly in a learned value net for real batched
        inference; the default here is a plain loop and carries none of the
        throughput benefit the batched call is meant to provide.
        """
        return [self.evaluate(dto) for dto in dtos]


class HeuristicValueFunction(ValueModel):
    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        self._weights = dict(DEFAULT_WEIGHTS)
        if weights is not None:
            unknown = [name for name in weights if name not in DEFAULT_WEIGHTS]
            if unknown:
                names = ", ".join(sorted(repr(name) for name in unknown))
                raise ValueError(f"unknown heuristic weight(s): {names}")
            for name, raw_weight in weights.items():
                if (
                    isinstance(raw_weight, bool)
                    or not isinstance(raw_weight, (int, float))
                    or not math.isfinite(float(raw_weight))
                ):
                    raise ValueError(f"heuristic weight {name!r} must be a finite number")
                self._weights[name] = float(raw_weight)

    def evaluate(self, masked_emulator_dto: Mapping[str, Any]) -> float:
        outcome = _terminal_outcome(masked_emulator_dto)
        if outcome == "victory":
            return self._weights["victory_bonus"]
        if outcome == "defeat":
            return self._weights["defeat_penalty"]
        features = self._extract_features(masked_emulator_dto)
        return sum(self._weights.get(name, 0.0) * feature for name, feature in features.items())

    def _extract_features(self, dto: Mapping[str, Any]) -> dict[str, float]:
        hp = _num(dto.get("hp"))
        max_hp = max(1.0, _num(dto.get("maxHp"), default=1.0))
        block = max(0.0, _num(dto.get("block")))

        all_enemies = _mapping_sequence(dto, "enemies")
        enemies: list[tuple[int, Mapping[str, Any]]] = []
        for index, enemy in enumerate(all_enemies):
            is_alive = enemy.get("isAlive", True)
            if not isinstance(is_alive, bool):
                raise ValueError(f"heuristic input enemies[{index}].isAlive must be a boolean")
            if is_alive:
                enemies.append((index, enemy))

        # Keep dead enemies in the denominator so killing a nearly-dead enemy cannot
        # make aggregate enemy_hp_ratio look worse merely by shrinking max HP.
        enemy_hp = sum(max(0.0, _num(enemy.get("hp"))) for enemy in all_enemies)
        enemy_max_hp = (
            sum(max(1.0, _num(enemy.get("maxHp"), default=1.0)) for enemy in all_enemies) or 1.0
        )

        incoming_before_block = 0.0
        for index, enemy in enemies:
            intent = enemy.get("intent")
            if intent is None:
                continue
            if not isinstance(intent, Mapping):
                raise ValueError(
                    f"heuristic input enemies[{index}].intent must be a mapping when provided"
                )
            damage = intent.get("attackDamage")
            if damage is None:
                continue
            repeats = max(0.0, _num(intent.get("attackRepeats"), default=1.0))
            incoming_before_block += max(0.0, _num(damage) * repeats)
        incoming = max(0.0, incoming_before_block - block)

        buff_debuff = 0.0
        for index, power in enumerate(_mapping_sequence(dto, "playerPowers")):
            power_type = power.get("type")
            if power_type is not None and not isinstance(power_type, str):
                raise ValueError(f"heuristic input playerPowers[{index}].type must be a string")
            sign = 1.0 if power_type == "Buff" else -1.0 if power_type == "Debuff" else 0.0
            buff_debuff += sign * _num(power.get("amount"))

        return {
            "player_hp_ratio": hp / max_hp,
            "player_block": block,
            "enemy_hp_ratio": enemy_hp / enemy_max_hp,
            "predicted_incoming_damage": incoming,
            "enemies_alive": float(len(enemies)),
            "buff_debuff_score": buff_debuff,
        }


def _terminal_outcome(dto: Mapping[str, Any]) -> str | None:
    outcome = dto.get("outcome")
    if outcome in ("victory", "defeat"):
        return outcome
    transition = dto.get("transition")
    if transition is not None and not isinstance(transition, Mapping):
        raise ValueError("heuristic input transition must be a mapping when provided")
    if isinstance(transition, Mapping) and transition.get("kind") == "combat_completed":
        victory = transition.get("victory")
        if victory is True:
            return "victory"
        if victory is False:
            return "defeat"
    return None


def _mapping_sequence(dto: Mapping[str, Any], field_name: str) -> list[Mapping[str, Any]]:
    value = dto.get(field_name)
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"heuristic input {field_name} must be a sequence of mappings")
    normalized: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"heuristic input {field_name}[{index}] must be a mapping")
        normalized.append(item)
    return normalized


def _num(value: Any, *, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("heuristic input numbers must be finite numeric values")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError("heuristic input numbers must be finite numeric values") from exc
    if not math.isfinite(number):
        raise ValueError("heuristic input numbers must be finite numeric values")
    return number
