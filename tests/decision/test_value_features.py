from __future__ import annotations

import copy
import unittest

from sts2_training.decision.value_features import (
    VALUE_FEATURE_NAMES,
    VALUE_FEATURE_SCHEMA_VERSION,
    combat_value_features,
)


def _card(
    card_id: str,
    type_: str,
    *,
    upgrade_level: int = 0,
    enchantment: dict | None = None,
    count: int | None = None,
    tinker_time_type: str | None = None,
) -> dict:
    card = {
        "id": card_id,
        "type": type_,
        "rarity": "Basic",
        "cost": 1,
        "targetType": "AnyEnemy" if type_ == "Attack" else "Self",
        "upgraded": upgrade_level > 0,
        "upgradeLevel": upgrade_level,
        "tinkerTimeType": tinker_time_type,
        "tinkerTimeRider": None,
        "enchantment": enchantment,
    }
    if count is not None:
        card["count"] = count
    return card


def _dto() -> dict:
    return {
        "mask_version": "1.2",
        "hp": 40,
        "maxHp": 80,
        "block": 5,
        "energy": 2,
        "enemies": [
            {
                "hp": 20,
                "maxHp": 40,
                "isAlive": True,
                "intent": {"attackDamage": 10, "attackRepeats": 2},
                "powers": [{"amount": 3}],
            }
        ],
        "hand": [
            _card("STRIKE", "Attack", upgrade_level=1),
            _card(
                "DEFEND",
                "Skill",
                enchantment={"id": "SHARP", "amount": 2, "status": "Normal"},
            ),
        ],
        "drawPile": [
            _card("BASH", "Attack", upgrade_level=2, count=3),
        ],
        "discardPile": [
            _card("DEFEND", "Skill", count=2, tinker_time_type="Alpha"),
        ],
        "exhaustPile": [_card("POWER", "Power", count=1)],
        "potions": [{"potion_id": "p"}],
        "playerPowers": [{"amount": 2}],
        "legal_actions": [
            {"action_id": "opaque-1", "action_type": "card"},
            {"action_id": "opaque-2", "action_type": "card"},
        ],
    }


class CombatValueFeaturesTest(unittest.TestCase):
    def test_feature_vector_is_stable_action_identity_free_and_count_weighted(self) -> None:
        vector = combat_value_features(_dto())
        by_name = dict(zip(VALUE_FEATURE_NAMES, vector, strict=True))

        self.assertEqual(VALUE_FEATURE_SCHEMA_VERSION, 2)
        self.assertEqual(len(vector), len(VALUE_FEATURE_NAMES))
        self.assertEqual(by_name["player_hp_ratio"], 0.5)
        self.assertEqual(by_name["player_block"], 5.0)
        self.assertEqual(by_name["energy"], 2.0)
        self.assertEqual(by_name["enemy_hp_ratio"], 0.5)
        self.assertEqual(by_name["incoming_damage"], 15.0)
        self.assertEqual(by_name["hand_size"], 2.0)
        self.assertEqual(by_name["draw_pile_size"], 3.0)
        self.assertEqual(by_name["discard_pile_size"], 2.0)
        self.assertEqual(by_name["exhaust_pile_size"], 1.0)
        self.assertEqual(by_name["known_card_count"], 8.0)
        self.assertEqual(by_name["upgraded_card_count"], 4.0)
        self.assertEqual(by_name["upgrade_level_sum"], 7.0)
        self.assertEqual(by_name["max_upgrade_level"], 2.0)
        self.assertEqual(by_name["enchanted_card_count"], 1.0)
        self.assertEqual(by_name["enchantment_amount_sum"], 2.0)
        self.assertEqual(by_name["tinker_time_card_count"], 2.0)
        self.assertEqual(by_name["attack_card_count"], 4.0)
        self.assertEqual(by_name["skill_card_count"], 3.0)
        self.assertEqual(by_name["power_card_count"], 1.0)
        self.assertEqual(by_name["upgraded_attack_count"], 4.0)
        self.assertEqual(by_name["enchanted_skill_count"], 1.0)
        self.assertEqual(by_name["hand_upgraded_card_count"], 1.0)
        self.assertEqual(by_name["hand_enchanted_card_count"], 1.0)
        self.assertEqual(by_name["potion_count"], 1.0)
        self.assertNotIn("action_id", VALUE_FEATURE_NAMES)

    def test_upgrade_and_enchantment_change_value_features(self) -> None:
        plain = _dto()
        plain["hand"] = [_card("STRIKE", "Attack")]
        upgraded = copy.deepcopy(plain)
        upgraded["hand"][0]["upgraded"] = True
        upgraded["hand"][0]["upgradeLevel"] = 1
        enchanted = copy.deepcopy(plain)
        enchanted["hand"][0]["enchantment"] = {
            "id": "SHARP",
            "amount": 3,
            "status": "Normal",
        }

        self.assertNotEqual(combat_value_features(plain), combat_value_features(upgraded))
        self.assertNotEqual(combat_value_features(plain), combat_value_features(enchanted))

    def test_legacy_mask_and_legacy_pile_shape_fail_closed(self) -> None:
        legacy = _dto()
        legacy["mask_version"] = "1.1"
        with self.assertRaisesRegex(ValueError, "mask_version='1.2'"):
            combat_value_features(legacy)

        malformed = _dto()
        malformed["drawPile"] = {"BASH": 3}
        with self.assertRaisesRegex(ValueError, "drawPile must be a sequence"):
            combat_value_features(malformed)


if __name__ == "__main__":
    unittest.main()
