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
from tests.dto_test_helpers import (
    card,
    dto,
    dto_replace,
    enemy,
    intent,
    pending_choice,
    potion,
    power,
)


def _card(card_id: str, *, upgraded: bool = False) -> dict:
    return card(
        id=card_id,
        type="Attack",
        rarity="Basic",
        cost=1,
        upgraded=upgraded,
    )


def _multiset_record(
    card_id: str,
    count: int,
    *,
    upgraded: bool = False,
    upgrade_level: int = 0,
    enchantment: dict | None = None,
) -> dict:
    return card(
        id=card_id,
        type="Attack",
        rarity="Basic",
        cost=1,
        target_type="AnyEnemy",
        upgraded=upgraded,
        upgrade_level=upgrade_level,
        tinker_time_type=None,
        tinker_time_rider=None,
        enchantment=enchantment,
        count=count,
    )


def _combat_start_dto(**overrides) -> dict:
    state = dto(
        mask_version="1.2",
        character_id="IRONCLAD",
        hp=70,
        max_hp=80,
        block=0,
        energy=3,
        stars=None,
        current_room_type="CombatRoom",
        boundary="stable",
        turn_number=1,
        combat_round_number=1,
        step_index=12,
        pending_choice={},
        room_context={"column": 1, "row": 2},
        relics=[{"id": "BURNING_BLOOD"}],
        potions=[None, potion(id="FIRE_POTION"), None],
        player_powers=[
            power(id="STRENGTH_POWER", amount=999999999, type="Buff"),
            power(id="BUFFER_POWER", amount=999999999, type="Buff"),
            power(id="REGEN_POWER", amount=999999999, type="Buff"),
            power(id="VULNERABLE_POWER", amount=2, type="Debuff"),
        ],
        hand=[_card("STRIKE_IRONCLAD"), _card("BASH", upgraded=True)],
        draw_pile=[
            _multiset_record("DEFEND_IRONCLAD", 1),
            _multiset_record("STRIKE_IRONCLAD", 1),
            _multiset_record(
                "STRIKE_IRONCLAD",
                1,
                upgraded=True,
                upgrade_level=1,
                enchantment={"id": "SHARP", "amount": 1},
            ),
        ],
        discard_pile=[],
        exhaust_pile=[],
        orb_slots=None,
        orbs=[],
        enemies=[
            enemy(
                id="CULTIST",
                hp=40,
                max_hp=48,
                block=0,
                is_alive=True,
                slot_name="A",
                powers=[power(id="RITUAL_POWER", amount=3, type="Buff")],
            ),
            enemy(
                id="CULTIST",
                hp=0,
                max_hp=48,
                block=0,
                is_alive=False,
                slot_name="B",
                powers=[],
            ),
        ],
    )
    return dto_replace(state, **overrides)


class DtoToScenarioSpecTest(unittest.TestCase):
    def test_rejects_legacy_or_missing_mask_version(self) -> None:
        for mask_version in ("1.1", None, "invalid"):
            with self.subTest(mask_version=mask_version):
                with self.assertRaisesRegex(ValueError, r"mask_version exactly '1\.2'"):
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

    def test_preserves_enemy_intent_and_state_log(self) -> None:
        cultist = enemy(
            id="CULTIST",
            hp=40,
            max_hp=48,
            block=0,
            is_alive=True,
            slot_name="A",
            intent=intent(state_id="INCANTATION"),
            state_log=["ENTRY", "INCANTATION"],
            powers=[],
        )

        spec = dto_to_scenario_spec(_combat_start_dto(enemies=[cultist]), seed=1)

        assert spec is not None
        self.assertEqual(spec["enemies"][0]["forced_move"], "INCANTATION")
        self.assertEqual(spec["enemies"][0]["state_log"], ["ENTRY", "INCANTATION"])

    def test_returns_none_with_no_living_enemies(self) -> None:
        state = _combat_start_dto(enemies=[enemy(id="CULTIST", hp=0, is_alive=False)])
        self.assertIsNone(dto_to_scenario_spec(state, seed=1))

    def test_returns_none_with_a_live_pending_choice(self) -> None:
        state = _combat_start_dto(pending_choice=pending_choice(choice_type="discard"))
        self.assertIsNone(dto_to_scenario_spec(state, seed=1))

    def test_preserves_hand_card_upgrade_level_and_enchantment(self) -> None:
        enchanted_card = card(
            id="STRIKE_IRONCLAD",
            upgraded=True,
            upgrade_level=1,
            enchantment={"id": "SHARP", "amount": 2, "status": "Normal"},
        )
        state = _combat_start_dto(hand=[enchanted_card])

        spec = dto_to_scenario_spec(state, seed=1)

        assert spec is not None
        hand_cards = spec["extra"]["hand_cards"]
        self.assertEqual(
            hand_cards,
            [
                {
                    "card_id": "STRIKE_IRONCLAD",
                    "is_upgraded": True,
                    "upgrade_level": 1,
                    "enchantment": {"id": "SHARP", "amount": 2, "status": "Normal"},
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
        self.assertEqual(spec["hand"], [])

    def test_expands_multiset_piles_into_extra_cards_preserving_upgrade_state(self) -> None:
        spec = dto_to_scenario_spec(_combat_start_dto(), seed=1)

        assert spec is not None
        self.assertEqual(spec["draw_pile"], [])
        self.assertEqual(spec["discard_pile"], [])
        self.assertEqual(spec["exhaust_pile"], [])

        draw_pile_cards = spec["extra"]["draw_pile_cards"]
        self.assertEqual(
            draw_pile_cards,
            [
                {"card_id": "DEFEND_IRONCLAD", "is_upgraded": False, "upgrade_level": 0},
                {"card_id": "STRIKE_IRONCLAD", "is_upgraded": False, "upgrade_level": 0},
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
        fields["enemies"] = [EnemyScenario(**enemy_spec) for enemy_spec in fields["enemies"]]
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
                {
                    "event": "selection",
                    "received": {"masked_emulator_dto": dto(current_room_type="MapSelect")},
                },
                {"event": "selection", "received": {"masked_emulator_dto": _combat_start_dto()}},
                {
                    "event": "selection",
                    "received": {
                        "masked_emulator_dto": _combat_start_dto(turn_number=2, hp=60)
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

            specs = harvest_scenarios_from_jsonl(path, exclude_final_combat=False, rng=random.Random(0))

            self.assertEqual(len(specs), 2)

    def test_exclude_final_combat_drops_the_last_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = [
                {"event": "selection", "received": {"masked_emulator_dto": _combat_start_dto()}},
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
                        "masked_emulator_dto": _combat_start_dto(
                            room_context={"column": 3, "row": 4}
                        )
                    },
                },
            ]
            path = self._write_log(Path(tmp), records)

            self.assertFalse(is_completed_run_log(path))
            specs = harvest_scenarios_auto(path, rng=random.Random(0))
            self.assertEqual(len(specs), 1)


if __name__ == "__main__":
    unittest.main()
