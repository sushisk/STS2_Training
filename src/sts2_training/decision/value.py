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

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

JsonObject = Mapping[str, Any]

# victory_bonus/defeat_penalty must dominate every feature-based score (any win
# beats any non-terminal state, any loss is worse than any non-terminal state)
# so beam search never prefers a "healthier-looking" loss over a win, or a
# needlessly risky non-terminal state over a clean win.
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


class ValueModel(ABC):
    """Scores one `masked_emulator_dto`; higher is better for Training."""

    @abstractmethod
    def evaluate(self, masked_emulator_dto: Mapping[str, Any]) -> float:
        raise NotImplementedError

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
        if weights:
            self._weights.update(weights)

    def evaluate(self, masked_emulator_dto: Mapping[str, Any]) -> float:
        outcome = _terminal_outcome(masked_emulator_dto)
        if outcome == "victory":
            return self._weights["victory_bonus"]
        if outcome == "defeat":
            return self._weights["defeat_penalty"]
        features = self._extract_features(masked_emulator_dto)
        return sum(self._weights.get(name, 0.0) * value for name, value in features.items())

    def _extract_features(self, dto: Mapping[str, Any]) -> dict[str, float]:
        hp = _num(dto.get("hp"))
        max_hp = max(1.0, _num(dto.get("maxHp"), default=1.0))
        block = _num(dto.get("block"))

        enemies = [
            e for e in (dto.get("enemies") or []) if isinstance(e, Mapping) and e.get("isAlive", True)
        ]
        enemy_hp = sum(max(0.0, _num(e.get("hp"))) for e in enemies)
        enemy_max_hp = sum(max(1.0, _num(e.get("maxHp"), default=1.0)) for e in enemies) or 1.0

        incoming = 0.0
        for enemy in enemies:
            intent = enemy.get("intent") or {}
            if not isinstance(intent, Mapping):
                continue
            damage = intent.get("attackDamage")
            if damage is not None:
                repeats = intent.get("attackRepeats", 1) or 1
                incoming += max(0.0, _num(damage) * _num(repeats) - block)

        buff_debuff = 0.0
        for power in dto.get("playerPowers") or []:
            if not isinstance(power, Mapping):
                continue
            power_type = power.get("type")
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
    """Return the terminal verdict exposed by RL, when present."""
    outcome = dto.get("outcome")
    if outcome in ("victory", "defeat"):
        return outcome
    transition = dto.get("transition")
    if isinstance(transition, Mapping) and "victory" in transition:
        return "victory" if transition["victory"] else "defeat"
    return None


def _num(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default
