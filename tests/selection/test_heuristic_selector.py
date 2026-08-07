import random

import pytest

from sts2_training.selection import HeuristicCombatSelector, NoAvailableActionError


def _action(action_id: str, action_type: str, is_available: bool = True) -> dict:
    return {
        "action_id": action_id,
        "action_type": action_type,
        "is_available": is_available,
        "parameters": {},
    }


def test_select_prefers_card_over_other_categories() -> None:
    legal_actions = [
        _action("a-end", "end_turn"),
        _action("a-choice", "choice_card"),
        _action("a-card", "card"),
    ]
    selector = HeuristicCombatSelector(rng=random.Random(0))

    chosen = selector.select(legal_actions)

    assert chosen["action_type"] == "card"


def test_select_falls_back_to_choice_confirm_when_no_card() -> None:
    legal_actions = [
        _action("a-confirm", "choice_confirm"),
        _action("a-skip", "choice_skip"),
    ]
    selector = HeuristicCombatSelector(rng=random.Random(0))

    chosen = selector.select(legal_actions)

    assert chosen["action_type"] == "choice_confirm"


def test_select_falls_back_to_other_action_types() -> None:
    legal_actions = [_action("a-potion", "potion")]
    selector = HeuristicCombatSelector(rng=random.Random(0))

    chosen = selector.select(legal_actions)

    assert chosen["action_id"] == "a-potion"


def test_select_ignores_unavailable_actions() -> None:
    legal_actions = [
        _action("a-card-unavailable", "card", is_available=False),
        _action("a-card-available", "card"),
    ]
    selector = HeuristicCombatSelector(rng=random.Random(0))

    chosen = selector.select(legal_actions)

    assert chosen["action_id"] == "a-card-available"


def test_select_raises_when_nothing_available() -> None:
    legal_actions = [_action("a-card", "card", is_available=False)]
    selector = HeuristicCombatSelector(rng=random.Random(0))

    with pytest.raises(NoAvailableActionError):
        selector.select(legal_actions)
