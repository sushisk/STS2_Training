"""Semantic builders/accessors for Emulator DTO fixtures used by tests.

Tests outside this module should avoid spelling Emulator DTO wire keys directly.
When the DTO schema changes, update the key maps here and keep behavior-focused
pytest cases expressed in semantic names.

Wire/envelope contract tests may still assert protocol keys such as ``instance_id``
when those keys are the contract under test; this module is specifically for the
contents of ``masked_emulator_dto`` and nested public Emulator records.
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
    """Build a public enemy record using semantic field names."""

    return _encode(fields, ENEMY_KEYS)


def enemy_replace(value: Mapping[str, Any], **fields: Any) -> dict[str, Any]:
    return _replace(value, fields, ENEMY_KEYS)


def enemy_get(value: Mapping[str, Any], field: str) -> Any:
    return _get(value, field, ENEMY_KEYS)


def power(**fields: Any) -> dict[str, Any]:
    """Build a public power record using semantic field names."""

    return _encode(fields, POWER_KEYS)


def card(**fields: Any) -> dict[str, Any]:
    """Build a public card record using semantic field names."""

    return _encode(fields, CARD_KEYS)


def card_replace(value: Mapping[str, Any], **fields: Any) -> dict[str, Any]:
    return _replace(value, fields, CARD_KEYS)


def intent(**fields: Any) -> dict[str, Any]:
    """Build a public enemy intent record using semantic field names."""

    return _encode(fields, INTENT_KEYS)
