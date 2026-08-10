from __future__ import annotations

import copy
import unittest

from sts2_training.decision.value import DEFAULT_WEIGHTS, HeuristicValueFunction


# Producer-contract fixture derived from STS2_RL's canonical public DTO audit
# (`Outputs/reports/rl_dto_exposure_audit_20260803.md` §2.2): player Powers are
# `playerPowers[{id, amount, type}]` and enemy Powers are nested under
# `enemies[*].powers`. `STRENGTH_POWER` is a canonical v109 Power ID from
# `Common/ids/powers.json`.
_CANONICAL_MASKED_ENEMY_POWER_FIXTURE = {
    "hp": 50,
    "maxHp": 50,
    "block": 0,
    "playerPowers": [],
    "enemies": [
        {
            "index": 0,
            "id": "CALCIFIED_CULTIST",
            "hp": 48,
            "maxHp": 48,
            "block": 0,
            "isAlive": True,
            "powers": [
                {
                    "id": "STRENGTH_POWER",
                    "amount": 2,
                    "type": "Buff",
                }
            ],
            "intent": {"attackDamage": 6, "attackRepeats": 1},
            "stateLog": [],
        }
    ],
}


def _named_power_only_value_fn(*, power_values: dict[str, float]) -> HeuristicValueFunction:
    weights = {name: 0.0 for name in DEFAULT_WEIGHTS}
    weights["named_power_score"] = 1.0
    return HeuristicValueFunction(weights=weights, power_values=power_values)


class PowerValueRegressionTest(unittest.TestCase):
    def test_generic_power_amount_saturates_at_three(self) -> None:
        value_fn = HeuristicValueFunction()
        base = {"hp": 50, "maxHp": 50, "enemies": []}
        capped = {
            **base,
            "playerPowers": [
                {"id": "UNKNOWN_DURATION_POWER", "type": "Buff", "amount": 3}
            ],
        }
        huge = {
            **base,
            "playerPowers": [
                {"id": "UNKNOWN_DURATION_POWER", "type": "Buff", "amount": 999}
            ],
        }

        self.assertEqual(value_fn.evaluate(huge), value_fn.evaluate(capped))

    def test_named_power_semantics_keep_raw_amount_beyond_generic_cap(self) -> None:
        value_fn = _named_power_only_value_fn(power_values={"SCALING_ENGINE": 2.0})
        dto = {
            "playerPowers": [
                {"id": "SCALING_ENGINE", "type": "Buff", "amount": 10}
            ]
        }

        self.assertEqual(value_fn.evaluate(dto), 20.0)

    def test_canonical_enemy_powers_shape_affects_pruning_value(self) -> None:
        value_fn = HeuristicValueFunction()
        powered = copy.deepcopy(_CANONICAL_MASKED_ENEMY_POWER_FIXTURE)
        neutral = copy.deepcopy(_CANONICAL_MASKED_ENEMY_POWER_FIXTURE)
        neutral["enemies"][0]["powers"] = []

        self.assertLess(value_fn.evaluate(powered), value_fn.evaluate(neutral))


if __name__ == "__main__":
    unittest.main()
