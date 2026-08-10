from __future__ import annotations

import unittest

from sts2_training.decision.value import DEFAULT_WEIGHTS, HeuristicValueFunction


def _single_feature_weights(name: str, value: float = 1.0) -> dict[str, float]:
    weights = {key: 0.0 for key in DEFAULT_WEIGHTS}
    weights[name] = value
    return weights


class HeuristicValueFunctionTest(unittest.TestCase):
    def test_victory_dominates_everything(self) -> None:
        value_fn = HeuristicValueFunction()
        low_hp_but_won = {"outcome": "victory", "hp": 1, "maxHp": 100}
        full_hp_no_outcome = {"hp": 100, "maxHp": 100, "enemies": []}

        self.assertGreater(
            value_fn.evaluate(low_hp_but_won), value_fn.evaluate(full_hp_no_outcome)
        )

    def test_defeat_is_worse_than_any_nonterminal_state(self) -> None:
        value_fn = HeuristicValueFunction()
        defeat = {"outcome": "defeat", "hp": 0, "maxHp": 100}
        bad_nonterminal = {"hp": 1, "maxHp": 100, "enemies": [{"hp": 50, "maxHp": 50, "isAlive": True}]}

        self.assertLess(value_fn.evaluate(defeat), value_fn.evaluate(bad_nonterminal))

    def test_transition_victory_flag_is_honored(self) -> None:
        value_fn = HeuristicValueFunction()
        dto = {"transition": {"kind": "combat_completed", "victory": True}}

        self.assertEqual(value_fn.evaluate(dto), DEFAULT_WEIGHTS["victory_bonus"])

    def test_unknown_transition_victory_is_not_treated_as_defeat(self) -> None:
        value_fn = HeuristicValueFunction()
        dto = {"transition": {"kind": "combat_completed", "victory": None}}

        self.assertNotEqual(value_fn.evaluate(dto), DEFAULT_WEIGHTS["defeat_penalty"])

    def test_combat_terminal_outcome_is_honored(self) -> None:
        value_fn = HeuristicValueFunction()
        victory = {"legal_actions": [], "terminal": True, "outcome": "victory", "hp": 4, "maxHp": 80}
        defeat = {"legal_actions": [], "terminal": True, "outcome": "defeat", "hp": 0, "maxHp": 80}
        healthy_nonterminal = {"hp": 80, "maxHp": 80, "enemies": []}

        self.assertEqual(value_fn.evaluate(victory), DEFAULT_WEIGHTS["victory_bonus"])
        self.assertEqual(value_fn.evaluate(defeat), DEFAULT_WEIGHTS["defeat_penalty"])
        self.assertGreater(value_fn.evaluate(victory), value_fn.evaluate(healthy_nonterminal))

    def test_whole_run_terminal_outcome_is_honored(self) -> None:
        value_fn = HeuristicValueFunction()
        victory = {"run_terminal": True, "outcome": "victory"}
        defeat = {"run_terminal": True, "outcome": "defeat"}

        self.assertEqual(value_fn.evaluate(victory), DEFAULT_WEIGHTS["victory_bonus"])
        self.assertEqual(value_fn.evaluate(defeat), DEFAULT_WEIGHTS["defeat_penalty"])

    def test_higher_hp_ratio_scores_better(self) -> None:
        value_fn = HeuristicValueFunction()
        healthy = {"hp": 80, "maxHp": 100, "enemies": []}
        hurt = {"hp": 20, "maxHp": 100, "enemies": []}

        self.assertGreater(value_fn.evaluate(healthy), value_fn.evaluate(hurt))

    def test_dead_enemies_are_excluded_from_enemy_hp(self) -> None:
        value_fn = HeuristicValueFunction()
        dead_enemy = {
            "hp": 50,
            "maxHp": 50,
            "enemies": [{"hp": 0, "maxHp": 40, "isAlive": False}],
        }
        no_enemy = {"hp": 50, "maxHp": 50, "enemies": []}

        self.assertEqual(value_fn.evaluate(dead_enemy), value_fn.evaluate(no_enemy))

    def test_incoming_damage_reduced_by_block(self) -> None:
        value_fn = HeuristicValueFunction()
        base = {
            "hp": 50,
            "maxHp": 50,
            "block": 0,
            "enemies": [{"hp": 10, "maxHp": 10, "isAlive": True, "intent": {"attackDamage": 20, "attackRepeats": 1}}],
        }
        blocked = {**base, "block": 20}

        self.assertGreater(value_fn.evaluate(blocked), value_fn.evaluate(base))

    def test_block_is_consumed_once_across_multiple_enemy_attacks(self) -> None:
        value_fn = HeuristicValueFunction(
            weights=_single_feature_weights("predicted_incoming_damage", -1.0)
        )
        dto = {
            "hp": 50,
            "maxHp": 50,
            "block": 10,
            "enemies": [
                {"hp": 10, "maxHp": 10, "isAlive": True, "intent": {"attackDamage": 10}},
                {"hp": 10, "maxHp": 10, "isAlive": True, "intent": {"attackDamage": 10}},
            ],
        }

        self.assertEqual(value_fn.evaluate(dto), -10.0)

    def test_player_power_type_direction(self) -> None:
        value_fn = HeuristicValueFunction()
        base = {"hp": 50, "maxHp": 50, "enemies": []}
        buffed = {**base, "playerPowers": [{"type": "Buff", "amount": 2}]}
        debuffed = {**base, "playerPowers": [{"type": "Debuff", "amount": 2}]}

        self.assertGreater(value_fn.evaluate(buffed), value_fn.evaluate(base))
        self.assertLess(value_fn.evaluate(debuffed), value_fn.evaluate(base))

    def test_enemy_buff_is_bad_and_enemy_debuff_is_good(self) -> None:
        value_fn = HeuristicValueFunction()
        base_enemy = {"hp": 20, "maxHp": 20, "isAlive": True}
        neutral = {"hp": 50, "maxHp": 50, "enemies": [base_enemy]}
        buffed_enemy = {
            **neutral,
            "enemies": [{**base_enemy, "powers": [{"id": "BUFF", "type": "Buff", "amount": 2}]}],
        }
        debuffed_enemy = {
            **neutral,
            "enemies": [{**base_enemy, "powers": [{"id": "DEBUFF", "type": "Debuff", "amount": 2}]}],
        }

        self.assertLess(value_fn.evaluate(buffed_enemy), value_fn.evaluate(neutral))
        self.assertGreater(value_fn.evaluate(debuffed_enemy), value_fn.evaluate(neutral))

    def test_dead_enemy_powers_do_not_affect_value(self) -> None:
        value_fn = HeuristicValueFunction()
        base_enemy = {"hp": 0, "maxHp": 20, "isAlive": False}
        base = {"hp": 50, "maxHp": 50, "enemies": [base_enemy]}
        powered = {
            **base,
            "enemies": [{**base_enemy, "powers": [{"type": "Buff", "amount": 999}]}],
        }

        self.assertEqual(value_fn.evaluate(powered), value_fn.evaluate(base))

    def test_power_without_amount_counts_as_one_effective_stack(self) -> None:
        value_fn = HeuristicValueFunction()
        base = {"hp": 50, "maxHp": 50, "enemies": []}
        buffed = {**base, "playerPowers": [{"type": "Buff"}]}

        self.assertGreater(value_fn.evaluate(buffed), value_fn.evaluate(base))

    def test_generic_power_amount_is_capped(self) -> None:
        value_fn = HeuristicValueFunction()
        base = {"hp": 50, "maxHp": 50, "enemies": []}
        capped = {**base, "playerPowers": [{"type": "Buff", "amount": 3}]}
        huge = {**base, "playerPowers": [{"type": "Buff", "amount": 999}]}

        self.assertEqual(value_fn.evaluate(huge), value_fn.evaluate(capped))

    def test_named_power_values_add_semantic_adjustment(self) -> None:
        value_fn = HeuristicValueFunction(
            weights=_single_feature_weights("named_power_score"),
            power_values={"SCALING_ENGINE": 4.0, "DANGEROUS_ENEMY_AURA": 6.0},
        )
        dto = {
            "playerPowers": [
                {"id": "SCALING_ENGINE", "type": "Buff", "amount": 10},
                {"id": "UNKNOWN_POWER", "type": "Buff", "amount": 999},
            ],
            "enemies": [
                {
                    "hp": 20,
                    "maxHp": 20,
                    "isAlive": True,
                    "powers": [
                        {"power_id": "DANGEROUS_ENEMY_AURA", "type": "Buff", "amount": 1}
                    ],
                }
            ],
        }

        self.assertEqual(value_fn.evaluate(dto), 34.0)

    def test_missing_fields_do_not_raise(self) -> None:
        value_fn = HeuristicValueFunction()
        self.assertIsInstance(value_fn.evaluate({}), float)

    def test_malformed_nested_containers_fail_closed(self) -> None:
        value_fn = HeuristicValueFunction()
        malformed = (
            {"enemies": {}},
            {"enemies": [42]},
            {"enemies": [{"isAlive": "yes"}]},
            {"enemies": [{"isAlive": True, "intent": []}]},
            {"enemies": [{"isAlive": True, "powers": {}}]},
            {"enemies": [{"isAlive": True, "powers": [None]}]},
            {"playerPowers": {}},
            {"playerPowers": [None]},
            {"playerPowers": [{"type": 123, "amount": 1}]},
            {"playerPowers": [{"id": 123, "amount": 1}]},
            {"transition": []},
        )

        for dto in malformed:
            with self.subTest(dto=dto):
                with self.assertRaisesRegex(ValueError, "heuristic input"):
                    value_fn.evaluate(dto)

    def test_evaluate_batch_default_matches_looping_evaluate(self) -> None:
        value_fn = HeuristicValueFunction()
        dtos = [{"hp": 10, "maxHp": 10}, {"hp": 5, "maxHp": 10}]

        self.assertEqual(value_fn.evaluate_batch(dtos), [value_fn.evaluate(d) for d in dtos])

    def test_custom_weights_override_defaults(self) -> None:
        value_fn = HeuristicValueFunction(weights={"player_hp_ratio": 0.0})
        full_hp = {"hp": 100, "maxHp": 100, "enemies": []}
        no_hp = {"hp": 0, "maxHp": 100, "enemies": []}

        self.assertEqual(value_fn.evaluate(full_hp), value_fn.evaluate(no_hp))

    def test_invalid_power_value_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "power_values keys"):
            HeuristicValueFunction(power_values={"": 1.0})
        with self.assertRaisesRegex(ValueError, "power value"):
            HeuristicValueFunction(power_values={"X": float("nan")})


if __name__ == "__main__":
    unittest.main()
