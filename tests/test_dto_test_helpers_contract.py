from __future__ import annotations

import unittest

from tests.dto_test_helpers import (
    action,
    action_parameters,
    card,
    choice_semantics,
    dto,
    enemy,
    intent,
    pending_choice,
    potion,
    power,
    transition,
)


class DtoTestHelpersContractTest(unittest.TestCase):
    """The intentional wire-shape assertions for behavior-test DTO helpers."""

    def test_dto_maps_semantic_names_to_wire_keys(self) -> None:
        self.assertEqual(
            dto(max_hp=80, player_powers=[power(id="STRENGTH", amount=2)]),
            {
                "maxHp": 80,
                "playerPowers": [{"id": "STRENGTH", "amount": 2}],
            },
        )

    def test_enemy_maps_semantic_names_to_wire_keys(self) -> None:
        self.assertEqual(
            enemy(max_hp=48, is_alive=True, slot_name="A"),
            {"maxHp": 48, "isAlive": True, "slotName": "A"},
        )

    def test_power_maps_semantic_names_to_wire_keys(self) -> None:
        self.assertEqual(
            power(power_id="VULNERABLE_POWER", type="Debuff", amount=2),
            {"power_id": "VULNERABLE_POWER", "type": "Debuff", "amount": 2},
        )

    def test_potion_maps_semantic_names_to_wire_keys(self) -> None:
        self.assertEqual(potion(potion_id="FIRE_POTION"), {"potion_id": "FIRE_POTION"})

    def test_card_maps_semantic_names_to_wire_keys(self) -> None:
        self.assertEqual(
            card(id="STRIKE", target_type="AnyEnemy", upgrade_level=1),
            {"id": "STRIKE", "targetType": "AnyEnemy", "upgradeLevel": 1},
        )

    def test_intent_maps_semantic_names_to_wire_keys(self) -> None:
        self.assertEqual(
            intent(state_id="ATTACK", attack_damage=6, attack_repeats=2),
            {"stateId": "ATTACK", "attackDamage": 6, "attackRepeats": 2},
        )

    def test_pending_choice_maps_semantic_names_to_wire_keys(self) -> None:
        self.assertEqual(
            pending_choice(
                selected_count=0,
                selected_option_ids=[],
                semantics=choice_semantics(version=1, operation="gain"),
            ),
            {
                "selectedCount": 0,
                "selectedOptionIds": [],
                "choiceSemantics": {"version": 1, "operation": "gain"},
            },
        )

    def test_action_maps_semantic_names_to_wire_keys(self) -> None:
        self.assertEqual(
            action(
                id="opaque-1",
                type="card",
                parameters=action_parameters(
                    card_id="STRIKE",
                    target_type="AnyEnemy",
                    enemy_index=0,
                ),
            ),
            {
                "action_id": "opaque-1",
                "action_type": "card",
                "parameters": {
                    "cardId": "STRIKE",
                    "targetType": "AnyEnemy",
                    "enemyIndex": 0,
                },
            },
        )

    def test_transition_maps_semantic_names_to_wire_keys(self) -> None:
        self.assertEqual(
            transition(kind="combat_completed", victory=True),
            {"kind": "combat_completed", "victory": True},
        )

    def test_unknown_semantic_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(KeyError, "unknown semantic DTO fields"):
            dto(creature_instances=[])


if __name__ == "__main__":
    unittest.main()
