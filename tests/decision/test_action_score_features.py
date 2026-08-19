from __future__ import annotations

import unittest

from sts2_training.decision.action_score_features import (
    ACTION_SCORE_FEATURE_NAMES,
    ACTION_SCORE_FEATURE_SCHEMA_VERSION,
    combat_action_score_features,
)
from tests.dto_test_helpers import (
    action,
    action_parameters,
    card,
    choice_semantics,
    dto,
    dto_replace,
    enemy,
    intent,
    pending_choice,
)


def _card(card_id: str, *, upgrade_level: int = 0, enchantment: dict | None = None) -> dict:
    return card(
        id=card_id,
        type="Attack",
        rarity="Common",
        cost=1,
        target_type="AnyEnemy",
        upgraded=upgrade_level > 0,
        upgrade_level=upgrade_level,
        tinker_time_type=None,
        tinker_time_rider=None,
        enchantment=enchantment,
    )


def _dto() -> dict:
    return dto(
        mask_version="1.2",
        dto_version="emulator-test",
        hp=40,
        max_hp=80,
        block=5,
        energy=2,
        enemies=[
            enemy(
                index=0,
                hp=20,
                max_hp=40,
                block=8,
                is_alive=True,
                intent=intent(attack_damage=7, attack_repeats=2),
                powers=[],
            )
        ],
        hand=[
            _card(
                "STRIKE",
                upgrade_level=2,
                enchantment={"id": "SHARP", "amount": 3, "status": "Normal"},
            )
        ],
        draw_pile=[],
        discard_pile=[],
        exhaust_pile=[],
        potions=[],
        player_powers=[],
    )


def _action(action_id: str) -> dict:
    return action(
        id=action_id,
        type="card",
        parameters=action_parameters(
            card_id="STRIKE",
            cost=1,
            target_type="AnyEnemy",
            enemy_index=0,
        ),
    )


class CombatActionScoreFeaturesTest(unittest.TestCase):
    def test_features_reuse_board_and_candidate_card_metadata_without_opaque_id(self) -> None:
        vector = combat_action_score_features(_dto(), _action("opaque-a"))
        by_name = dict(zip(ACTION_SCORE_FEATURE_NAMES, vector, strict=True))

        self.assertEqual(ACTION_SCORE_FEATURE_SCHEMA_VERSION, 4)
        self.assertEqual(by_name["board_player_hp_ratio"], 0.5)
        self.assertEqual(by_name["action_card"], 1.0)
        self.assertEqual(by_name["cost"], 1.0)
        self.assertEqual(by_name["affordable"], 1.0)
        self.assertEqual(by_name["card_type_attack"], 1.0)
        self.assertEqual(by_name["card_rarity_common"], 1.0)
        self.assertEqual(by_name["card_upgrade_level"], 2.0)
        self.assertEqual(by_name["card_enchanted"], 1.0)
        self.assertEqual(by_name["card_enchantment_amount"], 3.0)
        self.assertEqual(by_name["target_single_enemy"], 1.0)
        self.assertEqual(by_name["target_hp_ratio"], 0.5)
        self.assertEqual(by_name["target_block_ratio"], 0.2)
        self.assertEqual(by_name["target_incoming_attack"], 14.0)
        self.assertEqual(by_name["context_player_hp_ratio_x_action_card"], 0.5)
        self.assertAlmostEqual(by_name["context_danger_ratio_x_action_card"], 0.225)
        self.assertAlmostEqual(by_name["context_danger_ratio_x_card_type_attack"], 0.225)
        self.assertFalse(
            any("action_id" in name or "card_id" in name for name in ACTION_SCORE_FEATURE_NAMES)
        )

    def test_choice_operation_interaction_survives_pairwise_delta(self) -> None:
        state = _dto()
        attack = card(
            id="ATTACK_OPTION",
            type="Attack",
            rarity="Common",
            cost=1,
            target_type="AnyEnemy",
            upgraded=False,
            upgrade_level=0,
            tinker_time_type=None,
            tinker_time_rider=None,
            enchantment=None,
            option_id="attack-option",
        )
        skill = card(
            id="SKILL_OPTION",
            type="Skill",
            rarity="Common",
            cost=1,
            target_type="AnyEnemy",
            upgraded=False,
            upgrade_level=0,
            tinker_time_type=None,
            tinker_time_rider=None,
            enchantment=None,
            option_id="skill-option",
        )
        state = dto_replace(
            state,
            pending_choice=pending_choice(
                selected_count=0,
                min_select=1,
                max_select=1,
                selected_option_ids=[],
                options=[attack, skill],
                semantics=choice_semantics(version=1, operation="gain"),
            ),
        )
        attack_action = action(
            id="choose-attack",
            type="choice_card",
            parameters=action_parameters(option_id="attack-option"),
        )
        skill_action = action(
            id="choose-skill",
            type="choice_card",
            parameters=action_parameters(option_id="skill-option"),
        )
        attack_features = dict(
            zip(
                ACTION_SCORE_FEATURE_NAMES,
                combat_action_score_features(state, attack_action),
                strict=True,
            )
        )
        skill_features = dict(
            zip(
                ACTION_SCORE_FEATURE_NAMES,
                combat_action_score_features(state, skill_action),
                strict=True,
            )
        )
        feature_name = "context_choice_op_gain_x_card_type_attack"
        self.assertEqual(attack_features[feature_name] - skill_features[feature_name], 1.0)

    def test_opaque_action_identity_does_not_change_features(self) -> None:
        self.assertEqual(
            combat_action_score_features(_dto(), _action("opaque-a")),
            combat_action_score_features(_dto(), _action("opaque-b")),
        )

    def test_legacy_mask_fails_closed(self) -> None:
        state = dto_replace(_dto(), mask_version="1.1")
        with self.assertRaisesRegex(ValueError, "mask_version='1.2'"):
            combat_action_score_features(state, _action("a"))


if __name__ == "__main__":
    unittest.main()
