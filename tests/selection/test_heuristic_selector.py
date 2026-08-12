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


def test_select_takes_explicit_empty_slot_potion_reward() -> None:
    legal_actions = [
        _action("a-skip", "choice_reward_skip"),
        _action("a-take", "choice_reward_potion_take"),
    ]
    selector = HeuristicCombatSelector(rng=random.Random(0))

    chosen = selector.select(legal_actions)

    assert chosen["action_id"] == "a-take"


def test_select_skips_full_belt_potion_reward_instead_of_random_replace() -> None:
    legal_actions = [
        _action("a-replace-0", "choice_reward_potion_replace"),
        _action("a-replace-1", "choice_reward_potion_replace"),
        _action("a-skip", "choice_reward_skip"),
    ]
    selector = HeuristicCombatSelector(rng=random.Random(0))

    chosen = selector.select(legal_actions)

    assert chosen["action_id"] == "a-skip"


def test_select_replaces_when_reward_cannot_be_skipped() -> None:
    legal_actions = [
        _action("a-replace-0", "choice_reward_potion_replace"),
        _action("a-replace-1", "choice_reward_potion_replace"),
    ]
    selector = HeuristicCombatSelector(rng=random.Random(0))

    chosen = selector.select(legal_actions)

    assert chosen["action_type"] == "choice_reward_potion_replace"


def _card_action(action_id: str, card_id: str) -> dict:
    action = _action(action_id, "card")
    action["parameters"] = {"cardId": card_id}
    return action


def _dto_with_hand(*cards: dict) -> dict:
    return {"hp": 50, "energy": 3, "enemies": [], "hand": list(cards)}


def test_select_card_is_greedy_over_policy_score_when_epsilon_is_zero() -> None:
    legal_actions = [
        _card_action("a-curse", "ASCENDERS_BANE"),
        _card_action("a-attack", "STRIKE_IRONCLAD"),
    ]
    dto = _dto_with_hand(
        {"id": "ASCENDERS_BANE", "type": "Curse", "rarity": "Curse"},
        {"id": "STRIKE_IRONCLAD", "type": "Attack", "rarity": "Basic"},
    )
    selector = HeuristicCombatSelector(rng=random.Random(0), epsilon=0.0)

    chosen = selector.select(legal_actions, dto)

    assert chosen["action_id"] == "a-attack"


def test_select_card_explores_randomly_when_epsilon_is_one() -> None:
    legal_actions = [
        _card_action("a-curse", "ASCENDERS_BANE"),
        _card_action("a-attack", "STRIKE_IRONCLAD"),
    ]
    dto = _dto_with_hand(
        {"id": "ASCENDERS_BANE", "type": "Curse", "rarity": "Curse"},
        {"id": "STRIKE_IRONCLAD", "type": "Attack", "rarity": "Basic"},
    )
    selector = HeuristicCombatSelector(rng=random.Random(2), epsilon=1.0)

    chosen_ids = {selector.select(legal_actions, dto)["action_id"] for _ in range(20)}

    assert chosen_ids == {"a-curse", "a-attack"}


def _room_action(action_id: str, point_type: str) -> dict:
    action = _action(action_id, "map_room")
    action["parameters"] = {"column": 0, "row": 0, "point_type": point_type}
    return action


def test_select_room_is_greedy_over_room_score_when_epsilon_is_zero() -> None:
    legal_actions = [
        _room_action("a-elite", "Elite"),
        _room_action("a-treasure", "Treasure"),
    ]
    selector = HeuristicCombatSelector(rng=random.Random(0), epsilon=0.0)

    chosen = selector.select(legal_actions, {"hp": 80, "maxHp": 80})

    assert chosen["action_id"] == "a-treasure"


def test_select_room_prefers_rest_site_at_low_hp() -> None:
    legal_actions = [
        _room_action("a-monster", "Monster"),
        _room_action("a-rest", "RestSite"),
    ]
    selector = HeuristicCombatSelector(rng=random.Random(0), epsilon=0.0)

    chosen = selector.select(legal_actions, {"hp": 5, "maxHp": 80})

    assert chosen["action_id"] == "a-rest"


def test_select_room_explores_randomly_when_epsilon_is_one() -> None:
    legal_actions = [_room_action("a-elite", "Elite"), _room_action("a-treasure", "Treasure")]
    selector = HeuristicCombatSelector(rng=random.Random(3), epsilon=1.0)

    chosen_ids = {
        selector.select(legal_actions, {"hp": 80, "maxHp": 80})["action_id"] for _ in range(20)
    }

    assert chosen_ids == {"a-elite", "a-treasure"}
