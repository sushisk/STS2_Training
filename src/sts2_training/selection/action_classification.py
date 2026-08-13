"""Grouping of `masked_emulator_dto.legal_actions` entries by `action_type`.

Adapted from the OLD project's `choice_data.py` classification idea. Canonical live
mechanic meaning for card choices now lives in `pendingChoice.choiceSemantics` and must
be consumed through `choice_semantics.py`; this module intentionally remains structural
and never infers discard/exhaust/upgrade/etc. from prompts, card IDs, selector names, or
incidental keys.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

JsonObject = Mapping[str, Any]

CARD_ACTION_TYPE = "card"
CHOICE_CARD_ACTION_TYPE = "choice_card"
CHOICE_CONFIRM_ACTION_TYPE = "choice_confirm"
CHOICE_SKIP_ACTION_TYPE = "choice_skip"
CHOICE_TARGET_ACTION_TYPE = "choice_target"
CHOICE_REWARD_CARD_ACTION_TYPE = "choice_reward_card"
CHOICE_REWARD_POTION_TAKE_ACTION_TYPE = "choice_reward_potion_take"
CHOICE_REWARD_POTION_REPLACE_ACTION_TYPE = "choice_reward_potion_replace"
CHOICE_REWARD_SKIP_ACTION_TYPE = "choice_reward_skip"
MAP_ROOM_ACTION_TYPE = "map_room"
CHOICE_EVENT_OPTION_ACTION_TYPE = "choice_event_option"


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


def reward_card_actions(legal_actions: Sequence[JsonObject]) -> list[JsonObject]:
    return _actions_of_type(legal_actions, CHOICE_REWARD_CARD_ACTION_TYPE)


def reward_potion_take_actions(legal_actions: Sequence[JsonObject]) -> list[JsonObject]:
    return _actions_of_type(legal_actions, CHOICE_REWARD_POTION_TAKE_ACTION_TYPE)


def reward_potion_replace_actions(legal_actions: Sequence[JsonObject]) -> list[JsonObject]:
    return _actions_of_type(legal_actions, CHOICE_REWARD_POTION_REPLACE_ACTION_TYPE)


def reward_skip_actions(legal_actions: Sequence[JsonObject]) -> list[JsonObject]:
    return _actions_of_type(legal_actions, CHOICE_REWARD_SKIP_ACTION_TYPE)


def map_room_actions(legal_actions: Sequence[JsonObject]) -> list[JsonObject]:
    return _actions_of_type(legal_actions, MAP_ROOM_ACTION_TYPE)


def choice_event_option_actions(legal_actions: Sequence[JsonObject]) -> list[JsonObject]:
    return _actions_of_type(legal_actions, CHOICE_EVENT_OPTION_ACTION_TYPE)


def group_by_action_type(
    legal_actions: Sequence[JsonObject],
) -> dict[str, list[JsonObject]]:
    """Available actions grouped by `action_type`, preserving each group's order."""
    groups: dict[str, list[JsonObject]] = {}
    for action in available_actions(legal_actions):
        groups.setdefault(action.get("action_type"), []).append(action)
    return groups
