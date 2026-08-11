from __future__ import annotations

import random

from sts2_training.selection.choice_card_heuristic import choice_card_preference_scores
from sts2_training.selection.heuristic_selector import HeuristicCombatSelector


def _action(action_id: str, option_id: str) -> dict:
    return {
        "action_id": action_id,
        "action_type": "choice_card",
        "is_available": True,
        "parameters": {"optionId": option_id},
    }


def _dto(operation: str, options: list[dict], *, version: int = 1) -> dict:
    return {
        "pendingChoice": {
            "choiceSemantics": {
                "version": version,
                "operation": operation,
            },
            "options": options,
            "selectedOptionIds": [],
        }
    }


def test_discard_prefers_low_quality_card() -> None:
    actions = [_action("rare", "option-0"), _action("curse", "option-1")]
    dto = _dto(
        "discard",
        [
            {"id": "GOOD", "type": "Attack", "rarity": "Rare", "optionId": "option-0"},
            {"id": "BAD", "type": "Curse", "optionId": "option-1"},
        ],
    )

    scores = choice_card_preference_scores(actions, dto)

    assert scores["curse"] > scores["rare"]


def test_retrieve_prefers_high_quality_card() -> None:
    actions = [_action("curse", "option-0"), _action("rare", "option-1")]
    dto = _dto(
        "retrieve",
        [
            {"id": "BAD", "type": "Curse", "optionId": "option-0"},
            {"id": "GOOD", "type": "Attack", "rarity": "Rare", "optionId": "option-1"},
        ],
    )

    scores = choice_card_preference_scores(actions, dto)

    assert scores["rare"] > scores["curse"]


def test_duplicate_card_ids_are_ranked_by_option_identity() -> None:
    actions = [_action("plain", "option-0"), _action("upgraded", "option-1")]
    dto = _dto(
        "retrieve",
        [
            {
                "id": "STRIKE",
                "type": "Attack",
                "rarity": "Common",
                "upgraded": False,
                "optionId": "option-0",
            },
            {
                "id": "STRIKE",
                "type": "Attack",
                "rarity": "Common",
                "upgraded": True,
                "optionId": "option-1",
            },
        ],
    )

    scores = choice_card_preference_scores(actions, dto)

    assert scores["upgraded"] > scores["plain"]


def test_duplicate_action_option_ids_stay_neutral() -> None:
    actions = [_action("a", "option-0"), _action("b", "option-0")]
    dto = _dto(
        "retrieve",
        [
            {"id": "GOOD", "type": "Attack", "rarity": "Rare", "optionId": "option-0"},
            {"id": "BAD", "type": "Curse", "optionId": "option-1"},
        ],
    )

    assert choice_card_preference_scores(actions, dto) == {}


def test_unknown_future_or_inconsistent_choice_stays_neutral() -> None:
    actions = [_action("a", "option-0"), _action("b", "option-1")]
    options = [
        {"id": "A", "type": "Curse", "optionId": "option-0"},
        {"id": "B", "type": "Attack", "rarity": "Rare", "optionId": "option-1"},
    ]

    assert choice_card_preference_scores(actions, _dto("unknown", options)) == {}
    assert choice_card_preference_scores(actions, _dto("discard", options, version=2)) == {}

    mismatched = _dto("discard", options)
    mismatched["pendingChoice"]["options"][1]["optionId"] = "different-option"
    assert choice_card_preference_scores(actions, mismatched) == {}


def test_selector_uses_canonical_semantics() -> None:
    actions = [_action("rare", "option-0"), _action("curse", "option-1")]
    options = [
        {"id": "GOOD", "type": "Attack", "rarity": "Rare", "optionId": "option-0"},
        {"id": "BAD", "type": "Curse", "optionId": "option-1"},
    ]

    chosen = HeuristicCombatSelector(rng=random.Random(0)).select(
        actions, _dto("discard", options)
    )

    assert chosen["action_id"] == "curse"


def test_selector_without_dto_remains_backward_compatible() -> None:
    actions = [_action("a", "option-0"), _action("b", "option-1")]

    chosen = HeuristicCombatSelector(rng=random.Random(0)).select(actions)

    assert chosen in actions
