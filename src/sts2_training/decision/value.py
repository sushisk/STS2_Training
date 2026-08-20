"""`ValueModel`: scores resolved Combat states for global Beam pruning.

The ValueModel domain is deliberately **stable/terminal Combat state**, not an
interactive pending continuation. `BeamSearchEngine` resolves `choice_target`,
`choice_card`, `choice_confirm`, and `choice_skip` locally as part of the initiating
macro-action and only then calls `evaluate_batch`. A learned value model therefore does
not need a separate training target for partially resolved choice UI states.

`HeuristicValueFunction` and `PriorHeuristicPolicy` share `CombatObservation` for wire
normalization, so HP/block/enemy-intent/power interpretation cannot silently drift
between ranking and value code.

**Player survival is a single `effective_hp_ratio` term, not separate HP/block/incoming
terms.** Beam compares leaves that sit at different points in the turn cycle - one line
may have played a card and not yet taken the enemy's turn, while a sibling line has ended
the turn and already taken it. Any evaluator that scores raw HP and raw block separately
gives those leaves different amounts of not-yet-paid damage, so the comparison is decided
by turn parity rather than by play quality. Concretely, with the old
`player_hp_ratio`/`player_block`/`predicted_incoming_damage` split (40.0/+0.5/-1.0, and
`incoming_damage` already net of block), block was counted twice - once as itself and once
as damage it prevents - making one point of used block worth three points of HP, and
"end turn, then defend" outscored "defend, then end turn" even though both spend one card
and absorb one attack.

`effective_hp = hp - max(0, enemy intent total - block)` removes both problems at once. It
is turn-invariant: both orderings above evaluate to the same effective HP at equal depth,
while at shallower depth the line that blocks first is correctly ahead. Block earns exactly
the damage it actually prevents, so surplus block is worth nothing and block against a
non-attacking intent is worth nothing - which is correct, since block does not carry across
turns. Enemy intent is always published in the Combat DTO, so this never silently degrades
into "block has no value".

The term is deliberately not clamped at zero. A negative `effective_hp` means the published
enemy turn is lethal, and letting it go negative keeps a gradient toward reducing incoming
damage even in positions that are already losing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.decision.combat_observation import CombatObservation

JsonObject = Mapping[str, Any]

DEFAULT_WEIGHTS: dict[str, float] = {
    "effective_hp_ratio": 40.0,
    "enemy_hp_ratio": -30.0,
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

    ``exact_terminal_utility`` is a separate, opt-in contract for training labels. The
    default deliberately returns ``None``: a generic ValueModel prediction at a terminal
    DTO must not be treated as an uncensored terminal outcome merely because the state is
    terminal. Implementations may override it only when they can provide an exact utility
    in the same numeric scale as their value targets.

    ``oracle_provenance`` should return JSON-serializable metadata that distinguishes
    materially different model/checkpoint/configuration states when the model is used as
    an Oracle teacher. The default is empty for implementations without extra metadata.
    """

    def evaluate(self, masked_emulator_dto: Mapping[str, Any]) -> float:
        raise NotImplementedError("ValueModel must override evaluate or evaluate_batch")

    def evaluate_batch(self, dtos: Sequence[Mapping[str, Any]]) -> list[float]:
        return [self.evaluate(dto) for dto in dtos]

    def exact_terminal_utility(
        self, masked_emulator_dto: Mapping[str, Any]
    ) -> float | None:
        return None

    def oracle_provenance(self) -> Mapping[str, Any]:
        return {}


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

    def oracle_provenance(self) -> Mapping[str, Any]:
        return {
            "weights": dict(sorted(self._weights.items())),
            "power_values": dict(sorted(self._power_values.items())),
        }

    def evaluate(self, masked_emulator_dto: Mapping[str, Any]) -> float:
        terminal_utility = self.exact_terminal_utility(masked_emulator_dto)
        if terminal_utility is not None:
            return terminal_utility
        features = self._extract_features(masked_emulator_dto)
        return sum(
            self._weights.get(name, 0.0) * feature for name, feature in features.items()
        )

    def exact_terminal_utility(
        self, masked_emulator_dto: Mapping[str, Any]
    ) -> float | None:
        outcome = _terminal_outcome(masked_emulator_dto)
        if outcome == "victory":
            return self._weights["victory_bonus"]
        if outcome == "defeat":
            return self._weights["defeat_penalty"]
        return None

    def _extract_features(self, dto: Mapping[str, Any]) -> dict[str, float]:
        observation = CombatObservation.from_dto(dto, strict=True)
        all_enemies = observation.enemies
        alive_enemies = observation.alive_enemies

        enemy_hp = sum(enemy.hp for enemy in all_enemies)
        enemy_max_hp = sum(enemy.max_hp for enemy in all_enemies) or 1.0

        player_power_type_score = _typed_power_score(
            observation.player_powers, "playerPowers"
        )
        named_power_score = _named_power_score(
            observation.player_powers, self._power_values, "playerPowers"
        )

        enemy_power_type_score = 0.0
        for index, enemy in enumerate(alive_enemies):
            field_name = f"enemies[{index}].powers"
            enemy_power_type_score -= _typed_power_score(enemy.powers, field_name)
            named_power_score -= _named_power_score(
                enemy.powers, self._power_values, field_name
            )

        # `incoming_damage` is already `max(0, sum(enemy intents) - block)`, so this is the
        # HP the player is projected to hold once the enemy's published turn resolves.
        effective_hp = observation.hp - observation.incoming_damage
        return {
            "effective_hp_ratio": effective_hp / observation.max_hp,
            "enemy_hp_ratio": enemy_hp / enemy_max_hp,
            "enemies_alive": float(len(alive_enemies)),
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
    if outcome == "run_victory":
        return "victory"
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
