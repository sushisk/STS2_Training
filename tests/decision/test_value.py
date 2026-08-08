from __future__ import annotations

import unittest

from sts2_training.decision.value import DEFAULT_WEIGHTS, HeuristicValueFunction


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

    def test_real_combat_terminal_shape_from_rl_is_honored(self) -> None:
        # Exact shape STS2_RL's API/instance_combat.py::_decision_response_fields
        # sends at combat end (branch agent/expose-terminal-outcome) - `legal_actions`
        # is `[]`, `terminal`/`outcome` sit alongside it. Before that RL-side fix,
        # neither `terminal` nor `outcome` reached Training at all for a normal
        # (non-faulted) combat conclusion, so this exact shape never actually occurred
        # on the wire despite `_terminal_outcome` already checking for `outcome`.
        value_fn = HeuristicValueFunction()
        victory = {"legal_actions": [], "terminal": True, "outcome": "victory", "hp": 4, "maxHp": 80}
        defeat = {"legal_actions": [], "terminal": True, "outcome": "defeat", "hp": 0, "maxHp": 80}
        healthy_nonterminal = {"hp": 80, "maxHp": 80, "enemies": []}

        self.assertEqual(value_fn.evaluate(victory), DEFAULT_WEIGHTS["victory_bonus"])
        self.assertEqual(value_fn.evaluate(defeat), DEFAULT_WEIGHTS["defeat_penalty"])
        self.assertGreater(value_fn.evaluate(victory), value_fn.evaluate(healthy_nonterminal))

    def test_real_whole_run_terminal_shape_from_rl_is_honored(self) -> None:
        # Exact shape STS2_RL's API/instance_whole_run.py sends at RUN_TERMINAL
        # (branch agent/expose-terminal-outcome) - `run_terminal`/`outcome` together,
        # no `legal_actions` key at all. Before that fix the payload was hardcoded to
        # `{"run_terminal": True}` with no outcome, making a run victory and a run
        # defeat indistinguishable to this value function.
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

    def test_missing_fields_do_not_raise(self) -> None:
        value_fn = HeuristicValueFunction()
        self.assertIsInstance(value_fn.evaluate({}), float)

    def test_evaluate_batch_default_matches_looping_evaluate(self) -> None:
        value_fn = HeuristicValueFunction()
        dtos = [{"hp": 10, "maxHp": 10}, {"hp": 5, "maxHp": 10}]

        self.assertEqual(value_fn.evaluate_batch(dtos), [value_fn.evaluate(d) for d in dtos])

    def test_custom_weights_override_defaults(self) -> None:
        value_fn = HeuristicValueFunction(weights={"player_hp_ratio": 0.0})
        full_hp = {"hp": 100, "maxHp": 100, "enemies": []}
        no_hp = {"hp": 0, "maxHp": 100, "enemies": []}

        self.assertEqual(value_fn.evaluate(full_hp), value_fn.evaluate(no_hp))


if __name__ == "__main__":
    unittest.main()
