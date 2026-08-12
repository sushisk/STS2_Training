import random

from sts2_training.selection import (
    CHOICE_REWARD_CARD_ACTION_TYPE,
    HeuristicCombatSelector,
    RandomRewardCardSelectionPolicy,
    reward_card_actions,
)


def _action(action_id: str, action_type: str, *, is_available: bool = True) -> dict:
    return {
        "action_id": action_id,
        "action_type": action_type,
        "is_available": is_available,
        "parameters": {},
    }


def test_reward_card_actions_classifies_only_available_reward_cards() -> None:
    legal_actions = [
        _action("card-a", CHOICE_REWARD_CARD_ACTION_TYPE),
        _action("card-b", CHOICE_REWARD_CARD_ACTION_TYPE, is_available=False),
        _action("skip", "choice_reward_skip"),
    ]

    assert reward_card_actions(legal_actions) == [legal_actions[0]]


def test_random_reward_card_policy_preserves_historical_full_decision_randomness() -> None:
    legal_actions = [
        _action("card-a", CHOICE_REWARD_CARD_ACTION_TYPE),
        _action("card-b", CHOICE_REWARD_CARD_ACTION_TYPE),
        _action("card-c", CHOICE_REWARD_CARD_ACTION_TYPE),
        _action("skip", "choice_reward_skip"),
    ]
    policy = RandomRewardCardSelectionPolicy()
    rng = random.Random(0)

    chosen_ids = {
        policy.select(legal_actions, {"boundary": "reward_select"}, rng=rng)["action_id"]
        for _ in range(40)
    }

    assert chosen_ids == {"card-a", "card-b", "card-c", "skip"}


class _RecordingRewardCardPolicy:
    def __init__(self) -> None:
        self.action_ids: list[str] = []
        self.boundary = None

    def select(self, legal_actions, masked_emulator_dto, *, rng):
        del rng
        self.action_ids = [action["action_id"] for action in legal_actions]
        self.boundary = masked_emulator_dto.get("boundary")
        return next(action for action in legal_actions if action["action_id"] == "skip")


def test_heuristic_selector_delegates_reward_card_decision_to_replaceable_policy() -> None:
    legal_actions = [
        _action("card-a", CHOICE_REWARD_CARD_ACTION_TYPE),
        _action("card-unavailable", CHOICE_REWARD_CARD_ACTION_TYPE, is_available=False),
        _action("card-b", CHOICE_REWARD_CARD_ACTION_TYPE),
        _action("skip", "choice_reward_skip"),
    ]
    policy = _RecordingRewardCardPolicy()
    selector = HeuristicCombatSelector(
        rng=random.Random(0),
        reward_card_policy=policy,
    )

    chosen = selector.select(legal_actions, {"boundary": "reward_select"})

    assert chosen["action_id"] == "skip"
    assert policy.action_ids == ["card-a", "card-b", "skip"]
    assert policy.boundary == "reward_select"
