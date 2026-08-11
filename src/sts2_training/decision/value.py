"""`ValueModel`: scores resolved Combat states for global Beam pruning.

The ValueModel domain is deliberately **stable/terminal Combat state**, not an
interactive pending continuation. `BeamSearchEngine` resolves `choice_target`,
`choice_card`, `choice_confirm`, and `choice_skip` locally as part of the initiating
macro-action and only then calls `evaluate_batch`. A learned value model therefore does
not need a separate training target for partially resolved choice UI states.

`HeuristicValueFunction` is the no-model default: hand-picked features (HP/enemy-HP
ratios, block, predicted incoming damage, powers) combined through fixed weights.
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
    "enemy_buff_debuff_score": 2.0,
    "named_power_score": 1.0,
    "victory_bonus": 100_000.0,
    "defeat_penalty": -100_000.0,
}

_GENERIC_POWER_AMOUNT_CAP = 3.0
DEFAULT_POWER_VALUES: dict[str, float] = {}


class ValueModel:
    """Scores one resolved stable/terminal Combat DTO; higher is better.

    Pending interactive continuation DTOs are outside this interface's semantic domain.
    Beam resolves those locally before invoking the model. Implement `evaluate` for
    scalar inference or override `evaluate_batch` for real batched learned inference.
    """

    def evaluate(self, masked_emulator_dto: Mapping[str, Any]) -> float:
        raise NotImplementedError("ValueModel must override evaluate or evaluate_batch")

    def evaluate_batch(self, dtos: Sequence[Mapping[str, Any]]) -> list[float]:
        return [self.evaluate(dto) for dto in dtos]


class HeuristicValueFunction(ValueModel):
    """Hand-written ValueModel used for lightweight global beam pruning."""

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        *,
        power_values: Mapping[str, float] | None = None,
    ) -> None:
        self._weights = dict(DEFAULT_WEIGHTS)
        if weights is not None:
            unknown = [name for name in weights if name not in DEFAULT_WEIGHTS]
            if unknown:
                names = ", ".join(sorted(repr(name) for name in unknown))
                raise ValueError(f"unknown heuristic weight(s): {names}")
            for name, raw_weight in weights.items():
                self._weights[name] = _validated_finite_number(
                    raw_weight, f"heuristic weight {name!r}"
                )

        self._power_values = dict(DEFAULT_POWER_VALUES)
        if power_values is not None:
            for power_id, raw_value in power_values.items():
                if not isinstance(power_id, str) or not power_id:
                    raise ValueError("power_values keys must be non-empty strings")
                self._power_values[power_id] = _validated_finite_number(
                    raw_value, f"power value {power_id!r}"
                )

    def evaluate(self, masked_emulator_dto: Mapping[str, Any]) -> float:
        outcome = _terminal_outcome(masked_emulator_dto)
        if outcome == "victory":
            return self._weights["victory_bonus"]
        if outcome == "defeat":
            return self._weights["defeat_penalty"]
        features = self._extract_features(masked_emulator_dto)
        return sum(
            self._weights.get(name, 0.0) * feature
            for name, feature in features.items()
        )

    def _extract_features(self, dto: Mapping[str, Any]) -> dict[str, float]:
        hp = _num(dto.get("hp"))
        max_hp = max(1.0, _num(dto.get("maxHp"), default=1.0))
        block = max(0.0, _num(dto.get("block")))

        all_enemies = _mapping_sequence(dto, "enemies")
        enemies: list[tuple[int, Mapping[str, Any]]] = []
        for index, enemy in enumerate(all_enemies):
            is_alive = enemy.get("isAlive", True)
            if not isinstance(is_alive, bool):
                raise ValueError(
                    f"heuristic input enemies[{index}].isAlive must be a boolean"
                )
            if is_alive:
                enemies.append((index, enemy))

        enemy_hp = sum(max(0.0, _num(enemy.get("hp"))) for enemy in all_enemies)
        enemy_max_hp = (
            sum(
                max(1.0, _num(enemy.get("maxHp"), default=1.0))
                for enemy in all_enemies
            )
            or 1.0
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

        player_powers = _mapping_sequence(dto, "playerPowers")
        player_power_type_score = _typed_power_score(player_powers, "playerPowers")
        named_power_score = _named_power_score(
            player_powers, self._power_values, "playerPowers"
        )

        enemy_power_type_score = 0.0
        for index, enemy in enemies:
            raw_powers = enemy.get("powers")
            if raw_powers is None:
                raw_powers = enemy.get("enemyPowers")
            field_name = f"enemies[{index}].powers"
            enemy_powers = _mapping_sequence_value(raw_powers, field_name)
            enemy_power_type_score -= _typed_power_score(enemy_powers, field_name)
            named_power_score -= _named_power_score(
                enemy_powers, self._power_values, field_name
            )

        return {
            "player_hp_ratio": hp / max_hp,
            "player_block": block,
            "enemy_hp_ratio": enemy_hp / enemy_max_hp,
            "predicted_incoming_damage": incoming,
            "enemies_alive": float(len(enemies)),
            "buff_debuff_score": player_power_type_score,
            "enemy_buff_debuff_score": enemy_power_type_score,
            "named_power_score": named_power_score,
        }


def _typed_power_score(powers: Sequence[Mapping[str, Any]], field_name: str) -> float:
    score = 0.0
    for index, power in enumerate(powers):
        power_type = power.get("type")
        if power_type is not None and not isinstance(power_type, str):
            raise ValueError(f"heuristic input {field_name}[{index}].type must be a string")
        amount = min(
            abs(_num(power.get("amount"), default=1.0)),
            _GENERIC_POWER_AMOUNT_CAP,
        )
        if power_type == "Buff":
            score += amount
        elif power_type == "Debuff":
            score -= amount
    return score


def _named_power_score(
    powers: Sequence[Mapping[str, Any]],
    power_values: Mapping[str, float],
    field_name: str,
) -> float:
    score = 0.0
    for index, power in enumerate(powers):
        power_id = _power_id(power, f"{field_name}[{index}]")
        if power_id is None:
            continue
        value_per_stack = power_values.get(power_id)
        if value_per_stack is None:
            continue
        amount = abs(_num(power.get("amount"), default=1.0))
        score += value_per_stack * amount
    return score


def _power_id(power: Mapping[str, Any], field_name: str) -> str | None:
    for key in ("power_id", "powerId", "id"):
        value = power.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"heuristic input {field_name}.{key} must be a string")
        return value
    return None


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
    return _mapping_sequence_value(dto.get(field_name), field_name)


def _mapping_sequence_value(value: Any, field_name: str) -> list[Mapping[str, Any]]:
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


def _validated_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


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
