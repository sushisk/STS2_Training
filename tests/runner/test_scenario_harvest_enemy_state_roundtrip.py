from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from sts2_training.runner.oracle_collection import _scenario_from_json
from sts2_training.runner.scenario_harvest import (
    dto_to_scenario_spec,
    harvest_scenario_records_from_jsonl,
    is_completed_run_log,
    main,
)


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


class HarvestCompletionProvenanceTest(unittest.TestCase):
    def _combat_start_dto(self, *, column: int) -> dict:
        return {
            "mask_version": "1.2",
            "characterId": "IRONCLAD",
            "hp": 70,
            "maxHp": 80,
            "currentRoomType": "CombatRoom",
            "boundary": "stable",
            "turnNumber": 1,
            "combatRoundNumber": 1,
            "stepIndex": column,
            "pendingChoice": {},
            "room_context": {"column": column, "row": 1},
            "enemies": [
                {
                    "id": "CULTIST",
                    "hp": 40,
                    "maxHp": 48,
                    "isAlive": True,
                    "powers": [],
                }
            ],
        }

    def test_trailing_record_after_run_result_is_not_promotion_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            log_path = input_dir / "run.jsonl"
            records = [
                {
                    "event": "selection",
                    "received": {"masked_emulator_dto": self._combat_start_dto(column=1)},
                },
                {
                    "event": "selection",
                    "received": {"masked_emulator_dto": self._combat_start_dto(column=2)},
                },
                {
                    "event": "self_play_run_result",
                    "run_id": "stale-result",
                    "seed": 123,
                    "god_mode": False,
                    "outcome": "run_victory",
                },
                {
                    "event": "selection",
                    "received": {"masked_emulator_dto": {"currentRoomType": "MapSelect"}},
                },
            ]
            log_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            self.assertFalse(is_completed_run_log(log_path))
            harvested = harvest_scenario_records_from_jsonl(
                log_path,
                exclude_final_combat=True,
                rng=random.Random(0),
            )
            self.assertEqual(len(harvested), 1)
            self.assertFalse(harvested[0]["provenance"]["source_completed"])
            self.assertFalse(harvested[0]["provenance"]["promotion_eligible"])

            self.assertEqual(
                main(
                    [
                        "--input-dir",
                        str(input_dir),
                        "--output-dir",
                        str(output_dir),
                    ]
                ),
                0,
            )
            manifest = json.loads((output_dir / "harvest_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["scenarios"]), 1)
            self.assertFalse(manifest["scenarios"][0]["source_completed"])
            self.assertFalse(manifest["scenarios"][0]["promotion_eligible"])


if __name__ == "__main__":
    unittest.main()
