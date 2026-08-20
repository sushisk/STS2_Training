"""`DamageRaceValueFunction` behaviour, stated as the axioms it was designed to satisfy.

The three named cases are the ones the previous evaluators got wrong, and each is checked
here against `HeuristicValueFunction` so the difference is visible rather than asserted.
"""

from __future__ import annotations

import unittest

from sts2_training.decision.damage_race_value import DamageRaceValueFunction
from sts2_training.decision.value import HeuristicValueFunction

MAX_HP = 80
MAX_ENERGY = 3
BLOCK_PER_ENERGY = 5.0


def _enemy(hp: int, max_hp: int, damage: int | None = None, powers=None) -> dict:
    intent = (
        {"intentTypes": ["StatusCard"]}
        if damage is None
        else {"attackDamage": damage, "attackRepeats": 1, "intentTypes": ["Attack"]}
    )
    return {
        "hp": hp,
        "maxHp": max_hp,
        "isAlive": hp > 0,
        "block": 0,
        "powers": list(powers or []),
        "intent": intent,
    }


def _state(*, hp, block, energy, turn, enemies, max_energy=MAX_ENERGY) -> dict:
    return {
        "hp": hp,
        "maxHp": MAX_HP,
        "block": block,
        "energy": energy,
        "maxEnergy": max_energy,
        "turnNumber": turn,
        "combatRoundNumber": turn,
        "enemies": list(enemies),
        "legal_actions": [
            {"action_id": "0", "action_type": "system", "is_available": True}
        ],
    }


class ConversionNeutralityTest(unittest.TestCase):
    """Axiom: converting energy into block changes no value on its own."""

    def test_blocking_damage_that_would_land_anyway_is_neutral(self) -> None:
        value = DamageRaceValueFunction()
        enemies = [_enemy(30, 38, damage=12)]
        before = _state(hp=80, block=0, energy=3, turn=2, enemies=enemies)
        # One energy buys 5 block; the remaining energy still covers the rest of the hit.
        after = _state(hp=80, block=5, energy=2, turn=2, enemies=enemies)

        self.assertAlmostEqual(value.evaluate(after), value.evaluate(before), places=9)

    def test_block_against_a_non_attacking_enemy_costs_the_energy_it_used(self) -> None:
        """Not neutral: that energy could have shortened the fight instead."""

        value = DamageRaceValueFunction()
        enemies = [_enemy(30, 38)]
        before = _state(hp=80, block=0, energy=3, turn=2, enemies=enemies)
        after = _state(hp=80, block=5, energy=2, turn=2, enemies=enemies)

        self.assertLess(value.evaluate(after), value.evaluate(before))


class TurnPositionTest(unittest.TestCase):
    """The tie that made the search end turns with playable cards in hand."""

    def _tie_pair(self):
        enemies = [_enemy(30, 38, damage=12)]
        # A: Defend (5 block) then end the turn -> took 7, start of turn 2, 3 energy.
        a = _state(hp=73, block=0, energy=3, turn=2, enemies=enemies)
        # B: end the turn first -> took 12, then Defend -> mid turn 2, 2 energy.
        b = _state(hp=68, block=5, energy=2, turn=2, enemies=enemies)
        return a, b

    def test_the_old_evaluator_scores_the_two_orderings_identically(self) -> None:
        a, b = self._tie_pair()
        old = HeuristicValueFunction()

        self.assertAlmostEqual(old.evaluate(a), old.evaluate(b), places=9)

    def test_the_new_evaluator_prefers_blocking_first_by_the_hp_it_saved(self) -> None:
        a, b = self._tie_pair()
        value = DamageRaceValueFunction()

        # Exactly the 5 HP the earlier Defend actually prevented.
        self.assertAlmostEqual(value.evaluate(a) - value.evaluate(b), 5.0, places=9)


class EndTurnTest(unittest.TestCase):
    """Ending a turn must not profit from the energy it refills."""

    def test_ending_a_turn_into_an_incoming_attack_is_a_loss(self) -> None:
        value = DamageRaceValueFunction()
        enemies = [_enemy(30, 38, damage=12)]
        before = _state(hp=80, block=0, energy=3, turn=2, enemies=enemies)
        after = _state(hp=68, block=0, energy=3, turn=3, enemies=enemies)

        self.assertLess(value.evaluate(after), value.evaluate(before) - 12.0)

    def test_ending_a_turn_wastes_the_energy_it_did_not_spend(self) -> None:
        value = DamageRaceValueFunction()
        enemies = [_enemy(30, 38)]  # not attacking: nothing to block, no HP lost
        before = _state(hp=80, block=0, energy=3, turn=2, enemies=enemies)
        after = _state(hp=80, block=0, energy=3, turn=3, enemies=enemies)

        self.assertLess(value.evaluate(after), value.evaluate(before))

    def test_attacking_beats_blocking_beats_ending_against_a_non_attacking_enemy(self) -> None:
        value = DamageRaceValueFunction()
        base = [_enemy(30, 38)]
        strike = value.evaluate(_state(hp=80, block=0, energy=2, turn=2, enemies=[_enemy(24, 38)]))
        block = value.evaluate(_state(hp=80, block=5, energy=2, turn=2, enemies=base))
        end = value.evaluate(_state(hp=80, block=0, energy=3, turn=3, enemies=base))

        self.assertGreater(strike, block)
        self.assertGreater(block, end)


class CalibrationTest(unittest.TestCase):
    def test_damage_per_energy_ignores_the_current_turn(self) -> None:
        """Otherwise two leaves inside one turn get different exchange rates.

        Spending energy on block deals no damage, so charging the current turn to the
        denominator would lower the rate for whichever line happened to defend - and the
        conversion-neutrality axiom above would fail as a calibration artefact.
        """

        value = DamageRaceValueFunction()
        enemies = [_enemy(30, 38, damage=12)]
        spent = _state(hp=80, block=5, energy=2, turn=2, enemies=enemies)
        unspent = _state(hp=80, block=0, energy=3, turn=2, enemies=enemies)

        rate_spent = value._damage_per_energy(  # noqa: SLF001
            _observe(value, spent), float(MAX_ENERGY)
        )
        rate_unspent = value._damage_per_energy(  # noqa: SLF001
            _observe(value, unspent), float(MAX_ENERGY)
        )
        self.assertEqual(rate_spent, rate_unspent)

    def test_turn_one_uses_the_character_prior(self) -> None:
        value = DamageRaceValueFunction(default_damage_per_energy=6.0)
        state = _state(hp=80, block=0, energy=3, turn=1, enemies=[_enemy(38, 38, damage=12)])

        self.assertAlmostEqual(
            value._damage_per_energy(_observe(value, state), float(MAX_ENERGY)),  # noqa: SLF001
            6.0,
            places=9,
        )

    def test_a_harder_hitting_deck_calibrates_to_a_higher_rate(self) -> None:
        value = DamageRaceValueFunction()
        weak = _state(hp=80, block=0, energy=3, turn=3, enemies=[_enemy(32, 38, damage=12)])
        strong = _state(hp=80, block=0, energy=3, turn=3, enemies=[_enemy(8, 38, damage=12)])

        self.assertGreater(
            value._damage_per_energy(_observe(value, strong), float(MAX_ENERGY)),  # noqa: SLF001
            value._damage_per_energy(_observe(value, weak), float(MAX_ENERGY)),  # noqa: SLF001
        )


class MonotonicityTest(unittest.TestCase):
    def test_more_hp_is_better(self) -> None:
        value = DamageRaceValueFunction()
        enemies = [_enemy(30, 38, damage=12)]
        low = _state(hp=40, block=0, energy=3, turn=2, enemies=enemies)
        high = _state(hp=60, block=0, energy=3, turn=2, enemies=enemies)

        self.assertGreater(value.evaluate(high), value.evaluate(low))

    def test_less_enemy_hp_is_better(self) -> None:
        value = DamageRaceValueFunction()
        more = _state(hp=60, block=0, energy=3, turn=2, enemies=[_enemy(30, 38, damage=12)])
        less = _state(hp=60, block=0, energy=3, turn=2, enemies=[_enemy(10, 38, damage=12)])

        self.assertGreater(value.evaluate(less), value.evaluate(more))

    def test_a_cleared_board_costs_nothing_beyond_this_turn(self) -> None:
        value = DamageRaceValueFunction()
        state = _state(hp=60, block=0, energy=3, turn=2, enemies=[_enemy(0, 38, damage=12)])

        self.assertAlmostEqual(value.evaluate(state), 60.0, places=9)


class ContinuityTest(unittest.TestCase):
    def test_loss_is_continuous_where_full_blocking_stops_being_possible(self) -> None:
        """The two regimes of the closed form must meet at `enemy_damage == E * sigma`."""

        value = DamageRaceValueFunction()
        boundary = MAX_ENERGY * BLOCK_PER_ENERGY
        below = value._remaining_loss(  # noqa: SLF001
            remaining=30.0,
            enemy_damage=boundary - 1e-6,
            damage_per_energy=6.0,
            max_energy=float(MAX_ENERGY),
        )
        above = value._remaining_loss(  # noqa: SLF001
            remaining=30.0,
            enemy_damage=boundary + 1e-6,
            damage_per_energy=6.0,
            max_energy=float(MAX_ENERGY),
        )
        self.assertAlmostEqual(below, above, places=4)

    def test_loss_is_linear_in_remaining_enemy_hp(self) -> None:
        value = DamageRaceValueFunction()
        kwargs = {"enemy_damage": 12.0, "damage_per_energy": 6.0, "max_energy": float(MAX_ENERGY)}
        one = value._remaining_loss(remaining=10.0, **kwargs)  # noqa: SLF001
        two = value._remaining_loss(remaining=20.0, **kwargs)  # noqa: SLF001

        self.assertAlmostEqual(two, 2 * one, places=9)


class TerminalAndInputTest(unittest.TestCase):
    def test_terminal_outcomes_are_exact(self) -> None:
        value = DamageRaceValueFunction()

        self.assertEqual(value.evaluate({"outcome": "victory"}), 100_000.0)
        self.assertEqual(value.evaluate({"outcome": "defeat"}), -100_000.0)

    def test_missing_energy_fields_still_produce_a_finite_score(self) -> None:
        value = DamageRaceValueFunction()
        state = {"hp": 50, "maxHp": 80, "block": 0, "enemies": [_enemy(20, 38, damage=6)]}

        self.assertIsInstance(value.evaluate(state), float)

    def test_malformed_input_fails_closed(self) -> None:
        value = DamageRaceValueFunction()

        with self.assertRaisesRegex(ValueError, "heuristic input"):
            value.evaluate({"hp": 50, "maxHp": 80, "enemies": [42]})

    def test_invalid_parameters_are_rejected(self) -> None:
        for kwargs in (
            {"block_per_energy": 0.0},
            {"turn_attrition": -1.0},
            {"default_damage_per_energy": float("nan")},
            {"enemy_effective_hp_multipliers": {"": 1.0}},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    DamageRaceValueFunction(**kwargs)

    def test_provenance_names_the_model_and_its_parameters(self) -> None:
        provenance = DamageRaceValueFunction().oracle_provenance()

        self.assertEqual(provenance["model_type"], "damage_race_value_function")
        self.assertIn("block_per_energy", provenance)
        self.assertIn("turn_attrition", provenance)


class EnemyMultiplierTest(unittest.TestCase):
    def test_a_vulnerable_enemy_counts_as_less_remaining_hp(self) -> None:
        plain = DamageRaceValueFunction()
        aware = DamageRaceValueFunction(
            enemy_effective_hp_multipliers={"VULNERABLE_POWER": 1 / 1.5}
        )
        state = _state(
            hp=60,
            block=0,
            energy=3,
            turn=2,
            enemies=[_enemy(30, 38, damage=12, powers=[{"id": "VULNERABLE_POWER", "amount": 2}])],
        )

        self.assertGreater(aware.evaluate(state), plain.evaluate(state))

    def test_multipliers_are_off_by_default(self) -> None:
        self.assertEqual(
            DamageRaceValueFunction().oracle_provenance()["enemy_effective_hp_multipliers"], {}
        )


def _observe(value, dto):
    from sts2_training.decision.combat_observation import CombatObservation

    del value
    return CombatObservation.from_dto(dto, strict=True)


if __name__ == "__main__":
    unittest.main()
