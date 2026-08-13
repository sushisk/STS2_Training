from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from sts2_training.runner.scenario import CombatScenario, EnemyScenario
from sts2_training.runner.scenario_harvest import (
    dto_to_scenario_spec,
    harvest_scenarios_auto,
    harvest_scenarios_from_jsonl,
    is_completed_run_log,
)


def _card(card_id: str, *, upgraded: bool = False) -> dict:
    return {
        "id": card_id,
        "type": "Attack",
        "rarity": "Basic",
        "cost": 1,
        "upgraded": upgraded,
    }


def _multiset_record(card_id: str, count: int, **overrides) -> dict:
    record = {
        "id": card_id,
        "type": "Attack",
        "rarity": "Basic",
        "cost": 1,
        "targetType": "AnyEnemy",
        "upgraded": False,
        "upgradeLevel": 0,
        "tinkerTimeType": None,
        "tinkerTimeRider": None,
        "enchantment": None,
        "count": count,
    }
    record.update(overrides)
    return record


def _combat_start_dto(**overrides) -> dict:
    dto = {
        "mask_version": "1.2",
        "characterId": "IRONCLAD",
        "hp": 70,
        "maxHp": 80,
        "block": 0,
        "energy": 3,
        "stars": None,
        "currentRoomType": "CombatRoom",
        "boundary": "stable",
        "turnNumber": 1,
        "combatRoundNumber": 1,
        "stepIndex": 12,
        "pendingChoice": {},
        "room_context": {"column": 1, "row": 2},
        "relics": [{"id": "BURNING_BLOOD"}],
        "potions": [None, {"id": "FIRE_POTION"}, None],
        "playerPowers": [
            {"id": "STRENGTH_POWER", "amount": 999999999, "type": "Buff"},
            {"id": "BUFFER_POWER", "amount": 999999999, "type": "Buff"},
            {"id": "REGEN_POWER", "amount": 999999999, "type": "Buff"},
            {"id": "VULNERABLE_POWER", "amount": 2, "type": "Debuff"},
        ],
        "hand": [_card("STRIKE_IRONCLAD"), _card("BASH", upgraded=True)],
        "drawPile": [
            _multiset_record("DEFEND_IRONCLAD", 1),
            _multiset_record("STRIKE_IRONCLAD", 1),
            _multiset_record(
                "STRIKE_IRONCLAD",
                1,
                upgraded=True,
                upgradeLevel=1,
                enchantment={"id": "SHARP", "amount": 1},
            ),
        ],
        "discardPile": [],
        "exhaustPile": [],
        "orbSlots": None,
        "orbs": [],
        "enemies": [
            {
                "id": "CULTIST",
                "hp": 40,
                "maxHp": 48,
                "block": 0,
                "isAlive": True,
                "slotName": "A",
                "powers": [{"id": "RITUAL_POWER", "amount": 3, "type": "Buff"}],
            },
            {
                "id": "CULTIST",
                "hp": 0,
                "maxHp": 48,
                "block": 0,
                "isAlive": False,
                "slotName": "B",
                "powers": [],
            },
        ],
    }
    dto.update(overrides)
    return dto


class DtoToScenarioSpecTest(unittest.TestCase):
    def test_rejects_legacy_or_missing_mask_version(self) -> None:
        for mask_version in ("1.1", None, "invalid"):
            with self.subTest(mask_version=mask_version):
                with self.assertRaisesRegex(ValueError, r"mask_version >= 1\.2"):
                    dto_to_scenario_spec(_combat_start_dto(mask_version=mask_version), seed=1)

    def test_strips_god_mode_powers_but_keeps_others(self) -> None:
        spec = dto_to_scenario_spec(_combat_start_dto(), seed=1)

        assert spec is not None
        self.assertEqual(
            {power["power_id"] for power in spec["player_powers"]},
            {"VULNERABLE_POWER"},
        )

    def test_skips_unsafe_snapshots(self) -> None:
        unsafe = (
            _combat_start_dto(
                enemies=[{"id": "CULTIST", "hp": 0, "isAlive": False}]
            ),
            _combat_start_dto(pendingChoice={"choiceType": "discard"}),
        )
        for dto in unsafe:
            with self.subTest(dto=dto):
                self.assertIsNone(dto_to_scenario_spec(dto, seed=1))

    def test_preserves_hand_card_upgrade_level_and_enchantment(self) -> None:
        dto = _combat_start_dto(
            hand=[
                {
                    "id": "STRIKE_IRONCLAD",
                    "upgraded": True,
                    "upgradeLevel": 1,
                    "enchantment": {"id": "SHARP", "amount": 2, "status": "Normal"},
                }
            ]
        )

        spec = dto_to_scenario_spec(dto, seed=1)

        assert spec is not None
        self.assertEqual(spec["hand"], [])
        self.assertEqual(
            spec["extra"]["hand_cards"],
            [
                {
                    "card_id": "STRIKE_IRONCLAD",
                    "is_upgraded": True,
                    "upgrade_level": 1,
                    "enchantment": {"id": "SHARP", "amount": 2},
                }
            ],
        )

    def test_expands_multiset_piles_preserving_card_identity(self) -> None:
        spec = dto_to_scenario_spec(_combat_start_dto(), seed=1)

        assert spec is not None
        self.assertEqual(spec["draw_pile"], [])
        self.assertEqual(spec["discard_pile"], [])
        self.assertEqual(spec["exhaust_pile"], [])
        self.assertEqual(
            spec["extra"]["draw_pile_cards"],
            [
                {"card_id": "DEFEND_IRONCLAD", "is_upgraded": False},
                {"card_id": "STRIKE_IRONCLAD", "is_upgraded": False},
                {
                    "card_id": "STRIKE_IRONCLAD",
                    "is_upgraded": True,
                    "upgrade_level": 1,
                    "enchantment": {"id": "SHARP", "amount": 1},
                },
            ],
        )

    def test_produced_spec_deserializes_into_combat_scenario(self) -> None:
        spec = dto_to_scenario_spec(_combat_start_dto(), seed=42)
        assert spec is not None

        fields = dict(spec)
        fields["enemies"] = [EnemyScenario(**enemy) for enemy in fields["enemies"]]
        scenario = CombatScenario(**fields)
        instance_config = scenario.to_instance_config()

        self.assertEqual(scenario.character_id, "IRONCLAD")
        self.assertEqual(len(scenario.enemies), 1)
        self.assertEqual(instance_config["hand_cards"][1]["is_upgraded"], True)


class HarvestScenariosFromJsonlTest(unittest.TestCase):
    def _write_log(self, tmp: Path, records: list[dict]) -> Path:
        path = tmp / "run.jsonl"
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        return path

    def test_harvests_one_scenario_per_distinct_combat_room(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = [
                {"event": "self_play_run_result", "final_dto": _combat_start_dto()},
                {
                    "event": "selection",
                    "received": {"masked_emulator_dto": {"currentRoomType": "MapSelect"}},
                },
                {"event": "selection", "received": {"masked_emulator_dto": _combat_start_dto()}},
                {
                    "event": "selection",
                    "received": {
                        "masked_emulator_dto": _combat_start_dto(turnNumber=2, hp=60)
                    },
                },
                {
                    "event": "selection",
                    "received": {
                        "masked_emulator_dto": _combat_start_dto(
                            room_context={"column": 3, "row": 4}
                        )
                    },
                },
            ]
            path = self._write_log(Path(tmp), records)

            specs = harvest_scenarios_from_jsonl(
                path,
                exclude_final_combat=False,
                rng=random.Random(0),
            )

        self.assertEqual(len(specs), 2)

    def test_exclude_final_combat_drops_the_last_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_log(
                Path(tmp),
                [
                    {"event": "selection", "received": {"masked_emulator_dto": _combat_start_dto()}},
                    {
                        "event": "selection",
                        "received": {
                            "masked_emulator_dto": _combat_start_dto(
                                room_context={"column": 3, "row": 4}
                            )
                        },
                    },
                ],
            )
            specs = harvest_scenarios_from_jsonl(
                path,
                exclude_final_combat=True,
                rng=random.Random(0),
            )

        self.assertEqual(len(specs), 1)

    def test_auto_detects_completed_and_incomplete_runs(self) -> None:
        cases = (
            (
                "completed",
                [
                    {"event": "selection", "received": {"masked_emulator_dto": _combat_start_dto()}},
                    {"event": "self_play_run_result", "god_mode": True, "outcome": "run_victory"},
                ],
                True,
                1,
            ),
            (
                "incomplete",
                [
                    {"event": "selection", "received": {"masked_emulator_dto": _combat_start_dto()}},
                    {
                        "event": "selection",
                        "received": {
                            "masked_emulator_dto": _combat_start_dto(
                                room_context={"column": 3, "row": 4}
                            )
                        },
                    },
                ],
                False,
                1,
            ),
        )
        for name, records, completed, expected_count in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                path = self._write_log(Path(tmp), records)
                self.assertEqual(is_completed_run_log(path), completed)
                specs = harvest_scenarios_auto(path, rng=random.Random(0))
                self.assertEqual(len(specs), expected_count)


if __name__ == "__main__":
    unittest.main()
