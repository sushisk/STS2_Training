from __future__ import annotations

import unittest

from sts2_training.runner.scenario import CombatScenario, EnemyScenario, NewRunConfig, RunSnapshot


def _minimal_scenario(**overrides) -> CombatScenario:
    fields = dict(
        character_id="IRONCLAD",
        player_hp=50,
        player_max_hp=80,
        hand=["STRIKE_IRONCLAD"],
        draw_pile=[],
        discard_pile=[],
        enemies=[EnemyScenario(monster_id="CALCIFIED_CULTIST", hp=48)],
    )
    fields.update(overrides)
    return CombatScenario(**fields)


class CombatScenarioTest(unittest.TestCase):
    def test_missing_required_field_is_a_type_error(self) -> None:
        with self.assertRaises(TypeError):
            CombatScenario(character_id="IRONCLAD")  # type: ignore[call-arg]

    def test_empty_enemies_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _minimal_scenario(enemies=[])

    def test_extra_cannot_override_modeled_or_identity_fields(self) -> None:
        for key, value in (
            ("instance_type", "whole_run"),
            ("player_hp", 1),
            ("energy", 99),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                _minimal_scenario(extra={key: value})

    def test_to_instance_config_shape(self) -> None:
        scenario = _minimal_scenario()
        config = scenario.to_instance_config()

        self.assertEqual(config["instance_type"], "combat")
        self.assertEqual(config["character_id"], "IRONCLAD")
        self.assertEqual(config["player_hp"], 50)
        self.assertEqual(config["hand"], ["STRIKE_IRONCLAD"])
        self.assertEqual(config["enemies"], [{"monster_id": "CALCIFIED_CULTIST", "hp": 48}])
        self.assertEqual(config["potions"], [])
        self.assertEqual(config["player_powers"], [])
        self.assertNotIn("energy", config)
        self.assertNotIn("stars", config)

    def test_optional_numeric_fields_included_only_when_set(self) -> None:
        config = _minimal_scenario(energy=3, stars=2).to_instance_config()

        self.assertEqual(config["energy"], 3)
        self.assertEqual(config["stars"], 2)

    def test_extra_is_merged_verbatim(self) -> None:
        config = _minimal_scenario(extra={"pending_choice": {"foo": "bar"}}).to_instance_config()

        self.assertEqual(config["pending_choice"], {"foo": "bar"})

    def test_potion_ids_are_serialized_with_belt_slots(self) -> None:
        config = _minimal_scenario(potions=["FIRE_POTION", None, "BLOCK_POTION"]).to_instance_config()

        self.assertEqual(
            config["potions"],
            [
                {"slot": 0, "potion_id": "FIRE_POTION"},
                {"slot": 2, "potion_id": "BLOCK_POTION"},
            ],
        )

    def test_rl_shaped_potion_records_are_preserved(self) -> None:
        potion = {"slot": 2, "potion_id": "BLOCK_POTION"}

        config = _minimal_scenario(potions=[potion]).to_instance_config()

        self.assertEqual(config["potions"], [potion])
        self.assertIsNot(config["potions"][0], potion)

    def test_player_power_shorthand_is_serialized_to_stack_records(self) -> None:
        config = _minimal_scenario(player_powers={"STRENGTH": 2, "DEXTERITY": -1}).to_instance_config()

        self.assertEqual(
            config["player_powers"],
            [
                {"power_id": "STRENGTH", "amount": 2},
                {"power_id": "DEXTERITY", "amount": -1},
            ],
        )

    def test_exact_player_power_records_are_preserved(self) -> None:
        stack = {"power_id": "NIGHTMARE_POWER", "amount": 1, "associated_card": {"card_id": "STRIKE_IRONCLAD"}}

        config = _minimal_scenario(player_powers=[stack]).to_instance_config()

        self.assertEqual(config["player_powers"], [stack])
        self.assertIsNot(config["player_powers"][0], stack)

    def test_enemy_optional_fields_use_rl_wire_shape(self) -> None:
        enemy = EnemyScenario(
            monster_id="LOUSE_RED",
            hp=10,
            max_hp=12,
            block=3,
            slot_name="LEFT",
            frog_knight_has_beetle_charged=False,
            waterfall_giant_current_pressure_gun_damage=0,
            powers={"STRENGTH": 1},
        )

        self.assertEqual(
            enemy.to_dict(),
            {
                "monster_id": "LOUSE_RED",
                "hp": 10,
                "max_hp": 12,
                "block": 3,
                "slot_name": "LEFT",
                "frog_knight_has_beetle_charged": False,
                "waterfall_giant_current_pressure_gun_damage": 0,
                "powers": [{"power_id": "STRENGTH", "amount": 1}],
            },
        )

    def test_enemy_minimal_fields_only(self) -> None:
        enemy = EnemyScenario(monster_id="LOUSE_RED", hp=10)

        self.assertEqual(enemy.to_dict(), {"monster_id": "LOUSE_RED", "hp": 10})


class RunSnapshotTest(unittest.TestCase):
    def test_empty_snapshot_json_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RunSnapshot(character_id="IRONCLAD", ascension=0, seed=1, snapshot_json="")

    def test_to_instance_config_shape(self) -> None:
        snapshot = RunSnapshot(character_id="IRONCLAD", ascension=5, seed=42, snapshot_json="{...}")

        self.assertEqual(
            snapshot.to_instance_config(),
            {
                "instance_type": "whole_run",
                "character_id": "IRONCLAD",
                "ascension": 5,
                "seed": 42,
                "snapshot_json": "{...}",
            },
        )


class NewRunConfigTest(unittest.TestCase):
    def test_seed_omitted_when_none(self) -> None:
        config = NewRunConfig(character_id="IRONCLAD").to_instance_config()

        self.assertNotIn("seed", config)
        self.assertEqual(config["instance_type"], "whole_run")
        self.assertEqual(config["ascension"], 0)

    def test_seed_included_when_given(self) -> None:
        config = NewRunConfig(character_id="IRONCLAD", ascension=3, seed=7).to_instance_config()

        self.assertEqual(config["seed"], 7)
        self.assertEqual(config["ascension"], 3)


if __name__ == "__main__":
    unittest.main()
