from __future__ import annotations

from sts2_training.visualizer.page import INDEX_HTML
from sts2_training.visualizer.presentation import present_event


def test_presenter_preserves_max_hp_and_uses_choice_content() -> None:
    event = present_event(
        0,
        {
            "event": "selection",
            "received": {
                "decision_point_id": "d1",
                "masked_emulator_dto": {
                    "player_state": {"hp": 37, "maxHp": 80},
                    "monsters": [
                        {
                            "monsterName": "Jaw Worm",
                            "hp": 40,
                            "maxHp": 40,
                            "intent": {"move": "Chomp", "damage": 12},
                            "power": {"Strength": 2},
                            "currentBlock": 5,
                        }
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

    enemy = frame["enemies"][0]
    assert enemy["name"] == "Jaw Worm"
    assert enemy["intent"] == "Chomp"
    assert enemy["block"] == 5
    assert enemy["powers"][0]["name"] == "Strength"
    assert enemy["powers"][0]["amount"] == 2

    card_choice, system_choice = frame["choices"]
    assert card_choice["card_id"] == "BASH"
    assert card_choice["name"] == "BASH"
    assert card_choice["cost"] == 2
    assert system_choice["name"] == "end_turn"
    assert system_choice["name"] != system_choice["action_type"]


def test_enhanced_page_renders_enemy_facts_compact_card_choices_and_restart() -> None:
    assert "INTENT MOVE" in INDEX_HTML
    assert "enemy-facts" in INDEX_HTML
    assert "choice.card_id" in INDEX_HTML
    assert "actionType.includes('card')" in INDEX_HTML
    assert "['idle','completed','failed'].includes(data.state)" in INDEX_HTML
    assert "state.events=[];state.cursor=-1;state.frame=-1" in INDEX_HTML
