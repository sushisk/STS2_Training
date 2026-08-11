"""Grouping of `masked_emulator_dto.legal_actions` entries by `action_type`.

Adapted from the OLD project's `choice_data.py` classification idea, but scoped to
fields that the live API v0.5 response actually carries (see
`rl_training_dto_documentation.md`). The OLD `choice_meaning` categorization relied on
an offline `resolved.normalizedChoiceOperation` annotation produced by RL's teacher-data
export pipeline, which has no live equivalent here, so it is intentionally not ported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

JsonObject = Mapping[str, Any]

CARD_ACTION_TYPE = "card"
CHOICE_CARD_ACTION_TYPE = "choice_card"
CHOICE_CONFIRM_ACTION_TYPE = "choice_confirm"
CHOICE_SKIP_ACTION_TYPE = "choice_skip"
CHOICE_REWARD_POTION_TAKE_ACTION_TYPE = "choice_reward_potion_take"
CHOICE_REWARD_POTION_REPLACE_ACTION_TYPE = "choice_reward_potion_replace"
CHOICE_REWARD_SKIP_ACTION_TYPE = "choice_reward_skip"


def available_actions(legal_actions: Sequence[JsonObject]) -> list[JsonObject]:
    """Legal actions Training may select, excluding any explicitly marked unavailable."""
    return [a for a in legal_actions if a.get("is_available") is not False]


def _actions_of_type(
    legal_actions: Sequence[JsonObject], action_type: str
) -> list[JsonObject]:
    return [
        a for a in available_actions(legal_actions) if a.get("action_type") == action_type
    ]


def card_actions(legal_actions: Sequence[JsonObject]) -> list[JsonObject]:
    return _actions_of_type(legal_actions, CARD_ACTION_TYPE)


def choice_card_actions(legal_actions: Sequence[JsonObject]) -> list[JsonObject]:
    return _actions_of_type(legal_actions, CHOICE_CARD_ACTION_TYPE)


def choice_confirm_actions(legal_actions: Sequence[JsonObject]) -> list[JsonObject]:
    return _actions_of_type(legal_actions, CHOICE_CONFIRM_ACTION_TYPE)


def choice_skip_actions(legal_actions: Sequence[JsonObject]) -> list[JsonObject]:
    return _actions_of_type(legal_actions, CHOICE_SKIP_ACTION_TYPE)


def reward_potion_take_actions(legal_actions: Sequence[JsonObject]) -> list[JsonObject]:
    return _actions_of_type(legal_actions, CHOICE_REWARD_POTION_TAKE_ACTION_TYPE)


def reward_potion_replace_actions(legal_actions: Sequence[JsonObject]) -> list[JsonObject]:
    return _actions_of_type(legal_actions, CHOICE_REWARD_POTION_REPLACE_ACTION_TYPE)


def reward_skip_actions(legal_actions: Sequence[JsonObject]) -> list[JsonObject]:
    return _actions_of_type(legal_actions, CHOICE_REWARD_SKIP_ACTION_TYPE)


def group_by_action_type(
    legal_actions: Sequence[JsonObject],
) -> dict[str, list[JsonObject]]:
    """Available actions grouped by `action_type`, preserving each group's order."""
    groups: dict[str, list[JsonObject]] = {}
    for action in available_actions(legal_actions):
        groups.setdefault(action.get("action_type"), []).append(action)
    return groups
