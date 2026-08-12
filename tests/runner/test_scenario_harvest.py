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
    return {"id": card_id, "type": "Attack", "rarity": "Basic", "cost": 1, "upgraded": upgraded}


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
            {"id": "CULTIST", "hp": 0, "maxHp": 48, "block": 0, "isAlive": False, "slotName": "B", "powers": []},
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
        power_ids = {p["power_id"] for p in spec["player_powers"]}
        self.assertEqual(power_ids, {"VULNERABLE_POWER"})

    def test_drops_dead_enemies(self) -> None:
        spec = dto_to_scenario_spec(_combat_start_dto(), seed=1)

        assert spec is not None
        self.assertEqual(len(spec["enemies"]), 1)
        self.assertEqual(spec["enemies"][0]["monster_id"], "CULTIST")
        self.assertEqual(spec["enemies"][0]["hp"], 40)

    def test_returns_none_with_no_living_enemies(self) -> None:
        dto = _combat_start_dto(enemies=[{"id": "CULTIST", "hp": 0, "isAlive": False}])
        self.assertIsNone(dto_to_scenario_spec(dto, seed=1))

    def test_returns_none_with_a_live_pending_choice(self) -> None:
        dto = _combat_start_dto(pendingChoice={"choiceType": "discard"})
        self.assertIsNone(dto_to_scenario_spec(dto, seed=1))

    def test_preserves_hand_card_upgrade_level_and_enchantment(self) -> None:
        enchanted_card = {
            "id": "STRIKE_IRONCLAD",
            "upgraded": True,
            "upgradeLevel": 1,
            "enchantment": {"id": "SHARP", "amount": 2, "status": "Normal"},
        }
        dto = _combat_start_dto(hand=[enchanted_card])

        spec = dto_to_scenario_spec(dto, seed=1)

        assert spec is not None
        hand_cards = spec["extra"]["hand_cards"]
        self.assertEqual(
            hand_cards,
            [
                {
                    "card_id": "STRIKE_IRONCLAD",
                    "is_upgraded": True,
                    "upgrade_level": 1,
                    "enchantment": {"id": "SHARP", "amount": 2},
                }
            ],
        )

    def test_preserves_upgraded_cards_via_extra_hand_cards(self) -> None:
        spec = dto_to_scenario_spec(_combat_start_dto(), seed=1)

        assert spec is not None
        hand_cards = spec["extra"]["hand_cards"]
        self.assertEqual(
            hand_cards,
            [
                {"card_id": "STRIKE_IRONCLAD", "is_upgraded": False},
                {"card_id": "BASH", "is_upgraded": True},
            ],
        )
        # Plain piles stay empty - upgrade-preserving *_cards in extra carry the real data.
        self.assertEqual(spec["hand"], [])

    def test_expands_multiset_piles_into_extra_cards_preserving_upgrade_state(self) -> None:
        spec = dto_to_scenario_spec(_combat_start_dto(), seed=1)

        assert spec is not None
        # Plain draw_pile/discard_pile/exhaust_pile stay empty - the upgrade-preserving
        # *_cards extra fields carry the real data, mirroring hand_cards.
        self.assertEqual(spec["draw_pile"], [])
        self.assertEqual(spec["discard_pile"], [])
        self.assertEqual(spec["exhaust_pile"], [])

        draw_pile_cards = spec["extra"]["draw_pile_cards"]
        self.assertEqual(
            draw_pile_cards,
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
        self.assertNotIn("discard_pile_cards", spec["extra"])
        self.assertNotIn("exhaust_pile_cards", spec["extra"])

    def test_produced_spec_deserializes_into_a_real_combat_scenario(self) -> None:
        spec = dto_to_scenario_spec(_combat_start_dto(), seed=42)
        assert spec is not None

        fields = dict(spec)
        fields["enemies"] = [EnemyScenario(**enemy) for enemy in fields["enemies"]]
        scenario = CombatScenario(**fields)

        self.assertEqual(scenario.character_id, "IRONCLAD")
        self.assertEqual(len(scenario.enemies), 1)
        instance_config = scenario.to_instance_config()
        self.assertEqual(instance_config["hand_cards"][1]["is_upgraded"], True)


class HarvestScenariosFromJsonlTest(unittest.TestCase):
    def _write_log(self, tmp: Path, records: list[dict]) -> Path:
        path = tmp / "run.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        return path

    def test_harvests_one_scenario_per_distinct_combat_room(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = [
                {"event": "selection", "received": {"masked_emulator_dto": {"currentRoomType": "MapSelect"}}},
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
                        "masked_emulator_dto": _combat_start_dto(room_context={"column": 3, "row": 4})
                    },
                },
            ]
            path = self._write_log(Path(tmp), records)

            specs = harvest_scenarios_from_jsonl(path, exclude_final_combat=False, rng=random.Random(0))

            self.assertEqual(len(specs), 2)

    def test_exclude_final_combat_drops_the_last_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = [
                {"event": "selection", "received": {"masked_emulator_dto": _combat_start_dto()}},
                {
                    "event": "selection",
                    "received": {
                        "masked_emulator_dto": _combat_start_dto(room_context={"column": 3, "row": 4})
                    },
                },
            ]
            path = self._write_log(Path(tmp), records)

            specs = harvest_scenarios_from_jsonl(path, exclude_final_combat=True, rng=random.Random(0))

            self.assertEqual(len(specs), 1)

    def test_ignores_non_selection_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = [
                {"event": "self_play_run_result", "final_dto": _combat_start_dto()},
                {"event": "selection", "received": {"masked_emulator_dto": _combat_start_dto()}},
            ]
            path = self._write_log(Path(tmp), records)

            specs = harvest_scenarios_from_jsonl(path, exclude_final_combat=False, rng=random.Random(0))

            self.assertEqual(len(specs), 1)


class AutoDetectCompletionTest(unittest.TestCase):
    def _write_log(self, tmp: Path, records: list[dict]) -> Path:
        path = tmp / "run.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        return path

    def test_completed_run_keeps_the_final_combat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = [
                {"event": "selection", "received": {"masked_emulator_dto": _combat_start_dto()}},
                {"event": "self_play_run_result", "god_mode": True, "outcome": "run_victory"},
            ]
            path = self._write_log(Path(tmp), records)

            self.assertTrue(is_completed_run_log(path))
            specs = harvest_scenarios_auto(path, rng=random.Random(0))
            self.assertEqual(len(specs), 1)

    def test_incomplete_run_drops_the_final_combat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = [
                {"event": "selection", "received": {"masked_emulator_dto": _combat_start_dto()}},
                {
                    "event": "selection",
                    "received": {
                        "masked_emulator_dto": _combat_start_dto(room_context={"column": 3, "row": 4})
                    },
                },
            ]
            path = self._write_log(Path(tmp), records)

            self.assertFalse(is_completed_run_log(path))
            specs = harvest_scenarios_auto(path, rng=random.Random(0))
            self.assertEqual(len(specs), 1)


if __name__ == "__main__":
    unittest.main()
