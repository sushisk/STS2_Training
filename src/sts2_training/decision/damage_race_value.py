"""`DamageRaceValueFunction`: a Combat `ValueModel` measured in player HP.

The score is an estimate of **the HP the player will hold when this combat ends**, so a
mid-turn state and a start-of-turn state are directly comparable: both answer the same
question about the remainder of the fight. `HeuristicValueFunction`'s weighted sum cannot
do that, for two reasons that are visible in its own numbers.

*Its units depend on the encounter.* `enemy_hp_ratio` is worth `30 / sum(enemy maxHp)`
per enemy HP, so one point of enemy HP is 1.58x a point of player HP against a 38 HP
foe and 0.24x against a 250 HP boss. The exchange rate between the two resources the
fight is actually about swings by 6x with no one choosing it.

*It cannot express conversion.* Combat is a chain of conversions - energy into damage or
block, block into HP saved, enemy HP into fewer turns and therefore less damage taken - and
a sum of independent features has no notion of "spending x to get y leaves value
unchanged". Concretely, with `alpha = 40 / maxHp`, "Defend then end the turn" scores
`alpha*(H - I + b + beta - I')` and "end the turn then Defend" scores exactly the same,
because the previous fix (`effective_hp`) made the two turn-orderings algebraically
identical. The search then has nothing to choose between and `max()` keeps whichever came
first, which follows policy order. That tie is why the agent kept ending turns with
playable Strikes and Defends in hand: not a wrong preference, an absent one.

The tie is a real deficiency, not a tiebreak nuisance. Those two states are not equal -
one still holds a full turn of energy - and the old form charges the next turn's published
attack in full to a state that still has the resources to block it.

**The model.** Split each turn's energy budget `E` between attack `x` and defence `E-x`.
Then `x*kappa` damage per turn, `(E-x)*sigma` block per turn, and killing `R` enemy HP
takes `R/(x*kappa)` turns, each costing `max(0, D - (E-x)*sigma) + c` HP. A player picks
the split that hurts least, so the HP cost of the rest of the fight is the value of

    Loss(R, D) = min over 0 < x <= E of  R/(x*kappa) * (max(0, D - (E-x)*sigma) + c)

`c > 0` is the fixed per-turn attrition (chip damage, statuses, enemy scaling). Without it
"block forever and never attack" is a zero-cost solution and the minimum is degenerate.

That minimum has a closed form (`_remaining_loss`): the objective is piecewise monotone in
`x`, decreasing on both pieces, so the optimum is at the largest `x` that still blocks
everything, or at `x = E`. `Loss` is linear in `R`, and its slope is the exchange rate
between enemy HP and player HP - derived from `kappa`, `sigma`, `D`, `E`, `c` rather than
hand-set.

The current turn is not approximated: `I`, `b` and the remaining energy are all known, so
`_turn_cost` minimises the same objective exactly over the energy split. It is piecewise
linear, so only three splits can be optimal - spend nothing on block, spend exactly enough
to block, or spend everything - and all three are evaluated.

**No card table is required.** Card damage appears in neither the DTO (cards carry only
id/type/cost/upgrade) nor the bundled card export (`dmg_per_play` is 0.0 for all 439
entries). `kappa` is instead calibrated from the fight in progress - damage actually dealt
over the energy of the turns already completed, shrunk toward a character prior - so it
follows the deck, the upgrades and the character with no external data. See
`_damage_per_energy` for why the current turn is excluded from the denominator.

Energy is never scored on its own. It is valued only through the block and damage it can
still produce, and the state after ending a turn is scored by the same formula, so ending
a turn cannot profit from the energy it refills.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.decision.combat_observation import CombatObservation, EnemyObservation
from sts2_training.decision.value import ValueModel, _terminal_outcome

__all__ = ["DamageRaceValueFunction", "DEFAULT_TERMINAL_VALUES"]

DEFAULT_TERMINAL_VALUES: dict[str, float] = {
    "victory": 100_000.0,
    "defeat": -100_000.0,
}


class DamageRaceValueFunction(ValueModel):
    """Score a resolved Combat state as the HP the player is projected to end with.

    Every tunable is in player-HP-compatible units and there are only two hand-set
    numbers, `block_per_energy` and `turn_attrition`; `damage_per_energy` calibrates
    itself from the fight. `enemy_effective_hp_multipliers` is empty by default so the
    formula can be evaluated on its own before named-power adjustments are added.
    """

    def __init__(
        self,
        *,
        block_per_energy: float = 5.0,
        turn_attrition: float = 1.0,
        default_damage_per_energy: float = 6.0,
        min_damage_per_energy: float = 1.0,
        calibration_prior_turns: float = 1.0,
        enemy_effective_hp_multipliers: Mapping[str, float] | None = None,
        terminal_values: Mapping[str, float] | None = None,
    ) -> None:
        self._block_per_energy = _positive(block_per_energy, "block_per_energy")
        self._turn_attrition = _positive(turn_attrition, "turn_attrition")
        self._default_damage_per_energy = _positive(
            default_damage_per_energy, "default_damage_per_energy"
        )
        self._min_damage_per_energy = _positive(
            min_damage_per_energy, "min_damage_per_energy"
        )
        self._calibration_prior_turns = _positive(
            calibration_prior_turns, "calibration_prior_turns"
        )
        self._enemy_multipliers = dict(enemy_effective_hp_multipliers or {})
        for power_id, factor in self._enemy_multipliers.items():
            if not isinstance(power_id, str) or not power_id:
                raise ValueError("enemy_effective_hp_multipliers keys must be non-empty strings")
            _positive(factor, f"enemy_effective_hp_multiplier {power_id!r}")
        self._terminal_values = dict(DEFAULT_TERMINAL_VALUES)
        if terminal_values is not None:
            for outcome, value in terminal_values.items():
                self._terminal_values[outcome] = _finite(value, f"terminal value {outcome!r}")

    def oracle_provenance(self) -> Mapping[str, Any]:
        return {
            "model_type": "damage_race_value_function",
            "block_per_energy": self._block_per_energy,
            "turn_attrition": self._turn_attrition,
            "default_damage_per_energy": self._default_damage_per_energy,
            "min_damage_per_energy": self._min_damage_per_energy,
            "calibration_prior_turns": self._calibration_prior_turns,
            "enemy_effective_hp_multipliers": dict(sorted(self._enemy_multipliers.items())),
            "terminal_values": dict(sorted(self._terminal_values.items())),
        }

    def exact_terminal_utility(self, masked_emulator_dto: Mapping[str, Any]) -> float | None:
        outcome = _terminal_outcome(masked_emulator_dto)
        return None if outcome is None else self._terminal_values.get(outcome)

    def evaluate(self, masked_emulator_dto: Mapping[str, Any]) -> float:
        terminal = self.exact_terminal_utility(masked_emulator_dto)
        if terminal is not None:
            return terminal
        observation = CombatObservation.from_dto(masked_emulator_dto, strict=True)
        return observation.hp - self._turn_cost(observation)

    # -- the model -----------------------------------------------------------------

    def _turn_cost(self, observation: CombatObservation) -> float:
        """Minimum HP the rest of the fight costs, over how this turn's energy is split.

        Piecewise linear in the split, so only the two endpoints and the single kink -
        spending exactly enough energy to block everything - can be optimal.
        """

        alive = observation.alive_enemies
        remaining = self._effective_enemy_hp(alive)
        if remaining <= 0.0:
            # Nothing left to kill: only this turn's unblocked damage can still be paid.
            return max(0.0, observation.incoming_before_block - observation.block)

        max_energy = _budget(observation.max_energy, observation.energy)
        energy = max(0.0, observation.energy if observation.energy is not None else 0.0)
        damage_per_energy = self._damage_per_energy(observation, max_energy)
        incoming = observation.incoming_before_block
        threat = _unblocked_need(incoming, observation.block)

        exactly_enough = min(energy, threat / self._block_per_energy)
        candidates = {0.0, exactly_enough, energy}
        return min(
            self._cost_for_split(
                blocking_energy=split,
                energy=energy,
                threat=threat,
                remaining=remaining,
                damage_per_energy=damage_per_energy,
                max_energy=max_energy,
                incoming=incoming,
            )
            for split in candidates
        )

    def _cost_for_split(
        self,
        *,
        blocking_energy: float,
        energy: float,
        threat: float,
        remaining: float,
        damage_per_energy: float,
        max_energy: float,
        incoming: float,
    ) -> float:
        leak = max(0.0, threat - blocking_energy * self._block_per_energy)
        attacking_energy = max(0.0, energy - blocking_energy)
        left = max(0.0, remaining - attacking_energy * damage_per_energy)
        return leak + self._remaining_loss(
            remaining=left,
            enemy_damage=incoming,
            damage_per_energy=damage_per_energy,
            max_energy=max_energy,
        )

    def _remaining_loss(
        self,
        *,
        remaining: float,
        enemy_damage: float,
        damage_per_energy: float,
        max_energy: float,
    ) -> float:
        """Closed-form minimum of the damage race over the per-turn energy split.

        Two regimes. While a turn's energy can still absorb the whole incoming attack the
        only cost is attrition, minimised by attacking with every point of energy the
        block does not need. Once it cannot, blocking never pays for itself and the whole
        budget goes to ending the fight sooner. The two agree where they meet, so the
        result is continuous in `enemy_damage`.
        """

        if remaining <= 0.0:
            return 0.0
        race_rate = (enemy_damage + self._turn_attrition) / max_energy
        spare_energy = max_energy - enemy_damage / self._block_per_energy
        if spare_energy > 0.0:
            race_rate = min(race_rate, self._turn_attrition / spare_energy)
        return remaining * race_rate / damage_per_energy

    def _damage_per_energy(self, observation: CombatObservation, max_energy: float) -> float:
        """Damage this deck has actually produced per point of energy, this combat.

        Card damage is not published anywhere (see the module docstring), so the rate is
        read off the fight in progress. Before a turn has completed there is nothing to
        read and the character prior stands in.

        Two details keep the rate from becoming a source of incomparability itself.

        The denominator counts only **completed** turns, never the energy still unspent in
        the current one. Charging the current turn would make the rate depend on how this
        turn has gone so far - playing a Defend would lower it, because block spends energy
        without dealing damage - and two leaves inside the same turn would then be scored
        with different exchange rates. Measured: with the current turn included, the tie
        case scores +11.25 instead of +5 and playing a Defend stops being value-neutral,
        both purely as artefacts of the calibration rather than of the position.

        The estimate is then shrunk toward the character prior by
        `calibration_prior_turns` turns of pseudo-observation. One completed turn is a very
        small sample, and without shrinkage a single quiet turn halves the rate, which
        charges an end-of-turn twice: once for the HP and again for a suddenly slower race.
        Shrinkage also removes the need to special-case turn 1, where the estimate is
        simply the prior.
        """

        dealt = sum(max(0.0, enemy.max_hp - enemy.hp) for enemy in observation.enemies)
        turn = observation.turn_number
        completed = max(0, (turn - 1)) if turn is not None else 0
        observed_energy = max_energy * completed
        prior_energy = max_energy * self._calibration_prior_turns
        rate = (max(0.0, dealt) + prior_energy * self._default_damage_per_energy) / (
            observed_energy + prior_energy
        )
        return max(self._min_damage_per_energy, rate)

    def _effective_enemy_hp(self, alive: Sequence[EnemyObservation]) -> float:
        total = 0.0
        for enemy in alive:
            total += max(0.0, enemy.hp) + max(0.0, enemy.block)
        if not self._enemy_multipliers:
            return total
        adjusted = 0.0
        for enemy in alive:
            pool = max(0.0, enemy.hp) + max(0.0, enemy.block)
            adjusted += pool * self._multiplier_for(enemy)
        return adjusted

    def _multiplier_for(self, enemy: EnemyObservation) -> float:
        factor = 1.0
        for power in enemy.powers:
            if not isinstance(power, Mapping):
                continue
            for key in ("power_id", "powerId", "id"):
                power_id = power.get(key)
                if isinstance(power_id, str) and power_id in self._enemy_multipliers:
                    factor *= self._enemy_multipliers[power_id]
                    break
        return factor


def _unblocked_need(incoming: float, block: float) -> float:
    return max(0.0, incoming - max(0.0, block))


def _budget(max_energy: float | None, energy: float | None) -> float:
    """Per-turn energy budget, never zero.

    A DTO without `maxEnergy` still has to produce a finite score, so the current energy
    stands in and a floor of 1.0 keeps the rate finite.
    """

    for candidate in (max_energy, energy):
        if candidate is not None and candidate > 0.0:
            return float(candidate)
    return 1.0


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _positive(value: Any, label: str) -> float:
    number = _finite(value, label)
    if number <= 0.0:
        raise ValueError(f"{label} must be positive")
    return number
