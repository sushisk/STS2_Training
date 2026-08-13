from __future__ import annotations

import unittest

from sts2_training.runner.oracle_collection import _scenario_from_json
from sts2_training.runner.scenario_harvest import dto_to_scenario_spec


class HarvestedEnemyStateRoundTripTest(unittest.TestCase):
    def test_enemy_intent_and_state_log_survive_oracle_scenario_adapter(self) -> None:
        dto = {
            "mask_version": "1.2",
            "characterId": "IRONCLAD",
            "hp": 70,
            "maxHp": 80,
            "enemies": [
                {
                    "id": "CULTIST",
                    "hp": 40,
                    "maxHp": 48,
                    "isAlive": True,
                    "intent": {"stateId": "INCANTATION"},
                    "stateLog": ["ENTRY", "INCANTATION"],
                    "powers": [],
                }
            ],
        }

        spec = dto_to_scenario_spec(dto, seed=123)

        assert spec is not None
        scenario = _scenario_from_json(spec)
        enemy = scenario.to_instance_config()["enemies"][0]
        self.assertEqual(enemy["forced_move"], "INCANTATION")
        self.assertEqual(enemy["state_log"], ["ENTRY", "INCANTATION"])


if __name__ == "__main__":
    unittest.main()
