"""Semantic builders/accessors for Emulator DTO fixtures used by tests.

Behavior tests should avoid spelling Emulator DTO wire keys for records represented
by builders in this module. When those schema fields change, update the key maps
here and keep behavior-focused pytest cases expressed in semantic names.

``tests/test_dto_test_helpers_contract.py`` is the intentional exception: it pins
these semantic builders to the current wire contract. Wire/envelope contract tests
may likewise assert protocol keys such as ``instance_id`` when those keys are the
contract under test.

Small payloads without builders (for example room-context, relic, and enchantment
payload contents) are currently treated as opaque values. Add a builder before a
behavior test starts depending on their internal field names.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


DTO_KEYS: dict[str, str] = {
    "mask_version": "mask_version",
    "dto_version": "dto_version",
    "character_id": "characterId",
    "hp": "hp",
    "max_hp": "maxHp",
    "block": "block",
    "energy": "energy",
    "stars": "stars",
    "gold": "gold",
    "act_floor": "actFloor",
    "total_floor": "totalFloor",
    "current_room_type": "currentRoomType",
    "boundary": "boundary",
    "turn_number": "turnNumber",
    "combat_round_number": "combatRoundNumber",
    "step_index": "stepIndex",
    "pending_choice": "pendingChoice",
    "room_context": "room_context",
    "relics": "relics",
    "potions": "potions",
    "player_powers": "playerPowers",
    "hand": "hand",
    "deck": "deck",
    "draw_pile": "drawPile",
    "discard_pile": "discardPile",
    "exhaust_pile": "exhaustPile",
    "orb_slots": "orbSlots",
    "orbs": "orbs",
    "enemies": "enemies",
    "legal_actions": "legal_actions",
    "terminal": "terminal",
    "run_terminal": "run_terminal",
    "outcome": "outcome",
    "transition": "transition",
    "god_mode": "godMode",
}

ENEMY_KEYS: dict[str, str] = {
    "id": "id",
    "name": "name",
    "index": "index",
    "hp": "hp",
    "max_hp": "maxHp",
    "block": "block",
    "is_alive": "isAlive",
    "slot_name": "slotName",
    "intent": "intent",
    "state_log": "stateLog",
    "powers": "powers",
}

POWER_KEYS: dict[str, str] = {
    "id": "id",
    "power_id": "power_id",
    "amount": "amount",
    "type": "type",
}

POTION_KEYS: dict[str, str] = {
    "id": "id",
    "potion_id": "potion_id",
}

CARD_KEYS: dict[str, str] = {
    "id": "id",
    "type": "type",
    "rarity": "rarity",
    "cost": "cost",
    "target_type": "targetType",
    "upgraded": "upgraded",
    "upgrade_level": "upgradeLevel",
    "tinker_time_type": "tinkerTimeType",
    "tinker_time_rider": "tinkerTimeRider",
    "enchantment": "enchantment",
    "count": "count",
    "option_id": "optionId",
}

INTENT_KEYS: dict[str, str] = {
    "state_id": "stateId",
    "intent_types": "intentTypes",
    "attack_damage": "attackDamage",
    "attack_repeats": "attackRepeats",
}

PENDING_CHOICE_KEYS: dict[str, str] = {
    "choice_type": "choiceType",
    "selected_count": "selectedCount",
    "min_select": "minSelect",
    "max_select": "maxSelect",
    "selected_option_ids": "selectedOptionIds",
    "options": "options",
    "semantics": "choiceSemantics",
}

CHOICE_SEMANTICS_KEYS: dict[str, str] = {
    "version": "version",
    "operation": "operation",
}

ACTION_KEYS: dict[str, str] = {
    "id": "action_id",
    "type": "action_type",
    "available": "is_available",
    "parameters": "parameters",
}

ACTION_PARAMETER_KEYS: dict[str, str] = {
    "card_id": "cardId",
    "cost": "cost",
    "target_type": "targetType",
    "enemy_index": "enemyIndex",
    "option_id": "optionId",
    "command": "command",
}

TRANSITION_KEYS: dict[str, str] = {
    "kind": "kind",
    "victory": "victory",
}


def _encode(fields: Mapping[str, Any], keys: Mapping[str, str]) -> dict[str, Any]:
    unknown = set(fields) - set(keys)
    if unknown:
        raise KeyError(f"unknown semantic DTO fields: {sorted(unknown)!r}")
    return {keys[name]: copy.deepcopy(value) for name, value in fields.items()}


def _replace(
    value: Mapping[str, Any],
    fields: Mapping[str, Any],
    keys: Mapping[str, str],
) -> dict[str, Any]:
    replaced = copy.deepcopy(dict(value))
    replaced.update(_encode(fields, keys))
    return replaced


def _get(value: Mapping[str, Any], field: str, keys: Mapping[str, str]) -> Any:
    try:
        key = keys[field]
    except KeyError as exc:
        raise KeyError(f"unknown semantic DTO field: {field!r}") from exc
    return value[key]


def dto(**fields: Any) -> dict[str, Any]:
    """Build a (possibly partial) public Emulator DTO using semantic field names."""

    return _encode(fields, DTO_KEYS)


def dto_replace(value: Mapping[str, Any], **fields: Any) -> dict[str, Any]:
    """Deep-copy a DTO and replace fields by semantic name."""

    return _replace(value, fields, DTO_KEYS)


def dto_get(value: Mapping[str, Any], field: str) -> Any:
    """Read a DTO field by semantic name."""

    return _get(value, field, DTO_KEYS)


def enemy(**fields: Any) -> dict[str, Any]:
    return _encode(fields, ENEMY_KEYS)


def enemy_replace(value: Mapping[str, Any], **fields: Any) -> dict[str, Any]:
    return _replace(value, fields, ENEMY_KEYS)


def enemy_get(value: Mapping[str, Any], field: str) -> Any:
    return _get(value, field, ENEMY_KEYS)


def power(**fields: Any) -> dict[str, Any]:
    return _encode(fields, POWER_KEYS)


def potion(**fields: Any) -> dict[str, Any]:
    return _encode(fields, POTION_KEYS)


def card(**fields: Any) -> dict[str, Any]:
    return _encode(fields, CARD_KEYS)


def card_replace(value: Mapping[str, Any], **fields: Any) -> dict[str, Any]:
    return _replace(value, fields, CARD_KEYS)


def intent(**fields: Any) -> dict[str, Any]:
    return _encode(fields, INTENT_KEYS)


def pending_choice(**fields: Any) -> dict[str, Any]:
    return _encode(fields, PENDING_CHOICE_KEYS)


def choice_semantics(**fields: Any) -> dict[str, Any]:
    return _encode(fields, CHOICE_SEMANTICS_KEYS)


def action(**fields: Any) -> dict[str, Any]:
    return _encode(fields, ACTION_KEYS)


def action_parameters(**fields: Any) -> dict[str, Any]:
    return _encode(fields, ACTION_PARAMETER_KEYS)


def transition(**fields: Any) -> dict[str, Any]:
    return _encode(fields, TRANSITION_KEYS)
