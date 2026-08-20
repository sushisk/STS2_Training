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


def _event_action(action_id: str, will_kill_player) -> dict:
    action = _action(action_id, "choice_event_option")
    action["parameters"] = {"eventId": "SLIPPERY_BRIDGE", "choiceId": action_id, "willKillPlayer": will_kill_player}
    return action


def test_select_event_option_never_picks_a_confirmed_lethal_option_when_a_safe_one_exists() -> None:
    legal_actions = [_event_action("overcome", None), _event_action("hold_on", True)]
    selector = HeuristicCombatSelector(rng=random.Random(0), epsilon=1.0)

    # epsilon=1.0 (max exploration) still must never pick the lethal option - the
    # lethality filter is a hard constraint, not gated by epsilon like card/room are.
    chosen_ids = {selector.select(legal_actions)["action_id"] for _ in range(20)}

    assert chosen_ids == {"overcome"}


def test_select_event_option_allows_lethal_when_it_is_the_only_option() -> None:
    legal_actions = [_event_action("forced", True)]
    selector = HeuristicCombatSelector(rng=random.Random(0))

    chosen = selector.select(legal_actions)

    assert chosen["action_id"] == "forced"


def test_rest_options_use_the_rest_heuristic_instead_of_a_uniform_pick() -> None:
    """`choice_rest_option` had no branch and fell through to `_choose` (uniform random)."""

    selector = HeuristicCombatSelector(random.Random(0), epsilon=0.0)
    candidates = [
        {
            "action_id": "a-heal",
            "action_type": "choice_rest_option",
            "label": "HEAL",
            "is_available": True,
            "parameters": {"restOptionId": "HEAL"},
        },
        {
            "action_id": "a-smith",
            "action_type": "choice_rest_option",
            "label": "SMITH",
            "is_available": True,
            "parameters": {"restOptionId": "SMITH"},
        },
    ]

    hurt = selector.select(candidates, masked_emulator_dto={"hp": 8, "maxHp": 80})
    healthy = selector.select(candidates, masked_emulator_dto={"hp": 80, "maxHp": 80})

    assert hurt["action_id"] == "a-heal"
    assert healthy["action_id"] == "a-smith"


def test_rest_option_choice_is_deterministic_without_epsilon() -> None:
    candidates = [
        {
            "action_id": "a-heal",
            "action_type": "choice_rest_option",
            "label": "HEAL",
            "is_available": True,
            "parameters": {"restOptionId": "HEAL"},
        },
        {
            "action_id": "a-smith",
            "action_type": "choice_rest_option",
            "label": "SMITH",
            "is_available": True,
            "parameters": {"restOptionId": "SMITH"},
        },
    ]
    dto = {"hp": 8, "maxHp": 80}

    for seed in range(20):
        selector = HeuristicCombatSelector(random.Random(seed), epsilon=0.0)
        assert selector.select(candidates, masked_emulator_dto=dto)["action_id"] == "a-heal"
