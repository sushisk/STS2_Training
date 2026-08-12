from __future__ import annotations

from sts2_training.visualizer.dto_contract import present_event
from sts2_training.visualizer.page import INDEX_HTML


def test_presenter_prefers_formal_emulator_vitals_enemy_identity_and_intent() -> None:
    event = present_event(
        0,
        {
            "event": "selection",
            "received": {
                "decision_point_id": "d1",
                "masked_emulator_dto": {
                    "hp": 37,
                    "maxHp": 80,
                    "enemies": [
                        {
                            "id": "JAW_WORM",
                            "name": "Jaw Worm",
                            "hp": 40,
                            "maxHp": 44,
                            "block": 5,
                            "intent": {
                                "stateId": "CHOMP_MOVE",
                                "intentTypes": ["Attack"],
                                "attackDamage": 12,
                                "attackRepeats": 2,
                            },
                            "powers": [
                                {"id": "STRENGTH_POWER", "amount": 2, "type": "Buff"}
                            ],
                        },
                        {
                            "id": "CALCIFIED_CULTIST",
                            "name": "",
                            "hp": 18,
                            "maxHp": 48,
                            "block": 0,
                            "intent": {
                                "stateId": "INCANTATION_MOVE",
                                "intentTypes": ["Buff"],
                            },
                            "powers": [],
                        },
                    ],
                    "legal_actions": [
                        {
                            "action_id": "opaque-card-action",
                            "action_type": "card",
                            "parameters": {"cardId": "BASH", "cost": 2},
                        },
                        {
                            "action_id": "opaque-system-action",
                            "action_type": "system",
                            "parameters": {"command": "end_turn"},
                        },
                    ],
                },
            },
            "request": {"operation": "commit_action", "action_id": "opaque-card-action"},
        },
    )

    frame = event["frame"]
    assert frame["player"]["current_hp"] == 37
    assert frame["player"]["max_hp"] == 80

    attacking_enemy, unnamed_enemy = frame["enemies"]
    assert attacking_enemy["name"] == "Jaw Worm"
    assert attacking_enemy["current_hp"] == 40
    assert attacking_enemy["max_hp"] == 44
    assert attacking_enemy["intent"] == "CHOMP_MOVE · ATK 12×2"
    assert attacking_enemy["block"] == 5
    assert attacking_enemy["powers"][0]["id"] == "STRENGTH_POWER"
    assert attacking_enemy["powers"][0]["amount"] == 2

    # Emulator headless localization can produce an empty LogName. The model id is
    # still part of the same formal DTO, so it must win over a synthetic Enemy N.
    assert unnamed_enemy["name"] == "CALCIFIED_CULTIST"
    assert unnamed_enemy["current_hp"] == 18
    assert unnamed_enemy["max_hp"] == 48
    assert unnamed_enemy["intent"] == "INCANTATION_MOVE"

    card_choice, system_choice = frame["choices"]
    assert card_choice["card_id"] == "BASH"
    assert card_choice["name"] == "BASH"
    assert card_choice["cost"] == 2
    assert system_choice["name"] == "end_turn"
    assert system_choice["name"] != system_choice["action_type"]


def test_enhanced_page_renders_enemy_hp_intent_compact_card_choices_and_restart() -> None:
    assert "<span>HP</span>" in INDEX_HTML
    assert "hpText(value)" in INDEX_HTML
    assert "INTENT MOVE" in INDEX_HTML
    assert "enemy-facts" in INDEX_HTML
    assert "value.name || 'Enemy'" in INDEX_HTML
    assert "choice.card_id" in INDEX_HTML
    assert "actionType.includes('card')" in INDEX_HTML
    assert "['idle','completed','failed'].includes(data.state)" in INDEX_HTML
    assert "state.events=[];state.cursor=-1;state.frame=-1" in INDEX_HTML
