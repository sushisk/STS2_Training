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
        self.assertNotIn("energy", config)
        self.assertNotIn("stars", config)

    def test_optional_numeric_fields_included_only_when_set(self) -> None:
        config = _minimal_scenario(energy=3, stars=2).to_instance_config()

        self.assertEqual(config["energy"], 3)
        self.assertEqual(config["stars"], 2)

    def test_extra_is_merged_verbatim(self) -> None:
        config = _minimal_scenario(extra={"pending_choice": {"foo": "bar"}}).to_instance_config()

        self.assertEqual(config["pending_choice"], {"foo": "bar"})

    def test_enemy_optional_fields_included_only_when_set(self) -> None:
        enemy = EnemyScenario(monster_id="LOUSE_RED", hp=10, max_hp=12, block=3, powers={"STRENGTH": 1})

        self.assertEqual(
            enemy.to_dict(),
            {"monster_id": "LOUSE_RED", "hp": 10, "max_hp": 12, "block": 3, "powers": {"STRENGTH": 1}},
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
