from __future__ import annotations

import unittest

from sts2_training.decision.value import DEFAULT_WEIGHTS, HeuristicValueFunction
from tests.dto_test_helpers import dto, dto_replace, enemy, enemy_replace, intent, power


def _single_feature_weights(name: str, value: float = 1.0) -> dict[str, float]:
    weights = {key: 0.0 for key in DEFAULT_WEIGHTS}
    weights[name] = value
    return weights


class HeuristicValueFunctionTest(unittest.TestCase):
    def test_victory_dominates_everything(self) -> None:
        value_fn = HeuristicValueFunction()
        low_hp_but_won = dto(outcome="victory", hp=1, max_hp=100)
        full_hp_no_outcome = dto(hp=100, max_hp=100, enemies=[])

        self.assertGreater(
            value_fn.evaluate(low_hp_but_won), value_fn.evaluate(full_hp_no_outcome)
        )

    def test_defeat_is_worse_than_any_nonterminal_state(self) -> None:
        value_fn = HeuristicValueFunction()
        defeat = dto(outcome="defeat", hp=0, max_hp=100)
        bad_nonterminal = dto(
            hp=1,
            max_hp=100,
            enemies=[enemy(hp=50, max_hp=50, is_alive=True)],
        )

        self.assertLess(value_fn.evaluate(defeat), value_fn.evaluate(bad_nonterminal))

    def test_transition_victory_flag_is_honored(self) -> None:
        value_fn = HeuristicValueFunction()
        state = dto(transition={"kind": "combat_completed", "victory": True})

        self.assertEqual(value_fn.evaluate(state), DEFAULT_WEIGHTS["victory_bonus"])

    def test_unknown_transition_victory_is_not_treated_as_defeat(self) -> None:
        value_fn = HeuristicValueFunction()
        state = dto(transition={"kind": "combat_completed", "victory": None})

        self.assertNotEqual(value_fn.evaluate(state), DEFAULT_WEIGHTS["defeat_penalty"])

    def test_combat_terminal_outcome_is_honored(self) -> None:
        value_fn = HeuristicValueFunction()
        victory = dto(legal_actions=[], terminal=True, outcome="victory", hp=4, max_hp=80)
        defeat = dto(legal_actions=[], terminal=True, outcome="defeat", hp=0, max_hp=80)
        healthy_nonterminal = dto(hp=80, max_hp=80, enemies=[])

        self.assertEqual(value_fn.evaluate(victory), DEFAULT_WEIGHTS["victory_bonus"])
        self.assertEqual(value_fn.evaluate(defeat), DEFAULT_WEIGHTS["defeat_penalty"])
        self.assertGreater(value_fn.evaluate(victory), value_fn.evaluate(healthy_nonterminal))

    def test_whole_run_terminal_outcome_is_honored(self) -> None:
        value_fn = HeuristicValueFunction()
        victory = dto(run_terminal=True, outcome="victory")
        defeat = dto(run_terminal=True, outcome="defeat")

        self.assertEqual(value_fn.evaluate(victory), DEFAULT_WEIGHTS["victory_bonus"])
        self.assertEqual(value_fn.evaluate(defeat), DEFAULT_WEIGHTS["defeat_penalty"])

    def test_higher_hp_ratio_scores_better(self) -> None:
        value_fn = HeuristicValueFunction()
        healthy = dto(hp=80, max_hp=100, enemies=[])
        hurt = dto(hp=20, max_hp=100, enemies=[])

        self.assertGreater(value_fn.evaluate(healthy), value_fn.evaluate(hurt))

    def test_dead_enemies_are_excluded_from_enemy_hp(self) -> None:
        value_fn = HeuristicValueFunction()
        dead_enemy = dto(
            hp=50,
            max_hp=50,
            enemies=[enemy(hp=0, max_hp=40, is_alive=False)],
        )
        no_enemy = dto(hp=50, max_hp=50, enemies=[])

        self.assertEqual(value_fn.evaluate(dead_enemy), value_fn.evaluate(no_enemy))

    def test_incoming_damage_reduced_by_block(self) -> None:
        value_fn = HeuristicValueFunction()
        base = dto(
            hp=50,
            max_hp=50,
            block=0,
            enemies=[
                enemy(
                    hp=10,
                    max_hp=10,
                    is_alive=True,
                    intent=intent(attack_damage=20, attack_repeats=1),
                )
            ],
        )
        blocked = dto_replace(base, block=20)

        self.assertGreater(value_fn.evaluate(blocked), value_fn.evaluate(base))

    def test_block_is_consumed_once_across_multiple_enemy_attacks(self) -> None:
        value_fn = HeuristicValueFunction(
            weights=_single_feature_weights("predicted_incoming_damage", -1.0)
        )
        state = dto(
            hp=50,
            max_hp=50,
            block=10,
            enemies=[
                enemy(hp=10, max_hp=10, is_alive=True, intent=intent(attack_damage=10)),
                enemy(hp=10, max_hp=10, is_alive=True, intent=intent(attack_damage=10)),
            ],
        )

        self.assertEqual(value_fn.evaluate(state), -10.0)

    def test_player_power_type_direction(self) -> None:
        value_fn = HeuristicValueFunction()
        base = dto(hp=50, max_hp=50, enemies=[])
        buffed = dto_replace(base, player_powers=[power(type="Buff", amount=2)])
        debuffed = dto_replace(base, player_powers=[power(type="Debuff", amount=2)])

        self.assertGreater(value_fn.evaluate(buffed), value_fn.evaluate(base))
        self.assertLess(value_fn.evaluate(debuffed), value_fn.evaluate(base))

    def test_enemy_buff_is_bad_and_enemy_debuff_is_good(self) -> None:
        value_fn = HeuristicValueFunction()
        base_enemy = enemy(hp=20, max_hp=20, is_alive=True)
        neutral = dto(hp=50, max_hp=50, enemies=[base_enemy])
        buffed_enemy = dto_replace(
            neutral,
            enemies=[enemy_replace(base_enemy, powers=[power(id="BUFF", type="Buff", amount=2)])],
        )
        debuffed_enemy = dto_replace(
            neutral,
            enemies=[enemy_replace(base_enemy, powers=[power(id="DEBUFF", type="Debuff", amount=2)])],
        )

        self.assertLess(value_fn.evaluate(buffed_enemy), value_fn.evaluate(neutral))
        self.assertGreater(value_fn.evaluate(debuffed_enemy), value_fn.evaluate(neutral))

    def test_dead_enemy_powers_do_not_affect_value(self) -> None:
        value_fn = HeuristicValueFunction()
        base_enemy = enemy(hp=0, max_hp=20, is_alive=False)
        base = dto(hp=50, max_hp=50, enemies=[base_enemy])
        powered = dto_replace(
            base,
            enemies=[enemy_replace(base_enemy, powers=[power(type="Buff", amount=999)])],
        )

        self.assertEqual(value_fn.evaluate(powered), value_fn.evaluate(base))

    def test_power_without_amount_counts_as_one_effective_stack(self) -> None:
        value_fn = HeuristicValueFunction()
        base = dto(hp=50, max_hp=50, enemies=[])
        buffed = dto_replace(base, player_powers=[power(type="Buff")])

        self.assertGreater(value_fn.evaluate(buffed), value_fn.evaluate(base))

    def test_generic_power_amount_is_capped(self) -> None:
        value_fn = HeuristicValueFunction()
        base = dto(hp=50, max_hp=50, enemies=[])
        capped = dto_replace(base, player_powers=[power(type="Buff", amount=3)])
        huge = dto_replace(base, player_powers=[power(type="Buff", amount=999)])

        self.assertEqual(value_fn.evaluate(huge), value_fn.evaluate(capped))

    def test_named_power_values_add_semantic_adjustment(self) -> None:
        value_fn = HeuristicValueFunction(
            weights=_single_feature_weights("named_power_score"),
            power_values={"SCALING_ENGINE": 4.0, "DANGEROUS_ENEMY_AURA": 6.0},
        )
        state = dto(
            player_powers=[
                power(id="SCALING_ENGINE", type="Buff", amount=10),
                power(id="UNKNOWN_POWER", type="Buff", amount=999),
            ],
            enemies=[
                enemy(
                    hp=20,
                    max_hp=20,
                    is_alive=True,
                    powers=[
                        power(
                            power_id="DANGEROUS_ENEMY_AURA",
                            type="Buff",
                            amount=1,
                        )
                    ],
                )
            ],
        )

        self.assertEqual(value_fn.evaluate(state), 34.0)

    def test_missing_fields_do_not_raise(self) -> None:
        value_fn = HeuristicValueFunction()
        self.assertIsInstance(value_fn.evaluate(dto()), float)

    def test_malformed_nested_containers_fail_closed(self) -> None:
        value_fn = HeuristicValueFunction()
        malformed = (
            dto(enemies={}),
            dto(enemies=[42]),
            dto(enemies=[enemy(is_alive="yes")]),
            dto(enemies=[enemy(is_alive=True, intent=[])]),
            dto(enemies=[enemy(is_alive=True, powers={})]),
            dto(enemies=[enemy(is_alive=True, powers=[None])]),
            dto(player_powers={}),
            dto(player_powers=[None]),
            dto(player_powers=[power(type=123, amount=1)]),
            dto(player_powers=[power(id=123, amount=1)]),
            dto(transition=[]),
        )

        for state in malformed:
            with self.subTest(dto=state):
                with self.assertRaisesRegex(ValueError, "heuristic input"):
                    value_fn.evaluate(state)

    def test_evaluate_batch_default_matches_looping_evaluate(self) -> None:
        value_fn = HeuristicValueFunction()
        dtos = [dto(hp=10, max_hp=10), dto(hp=5, max_hp=10)]

        self.assertEqual(value_fn.evaluate_batch(dtos), [value_fn.evaluate(d) for d in dtos])

    def test_custom_weights_override_defaults(self) -> None:
        value_fn = HeuristicValueFunction(weights={"player_hp_ratio": 0.0})
        full_hp = dto(hp=100, max_hp=100, enemies=[])
        no_hp = dto(hp=0, max_hp=100, enemies=[])

        self.assertEqual(value_fn.evaluate(full_hp), value_fn.evaluate(no_hp))

    def test_invalid_power_value_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "power_values keys"):
            HeuristicValueFunction(power_values={"": 1.0})
        with self.assertRaisesRegex(ValueError, "power value"):
            HeuristicValueFunction(power_values={"X": float("nan")})


if __name__ == "__main__":
    unittest.main()
