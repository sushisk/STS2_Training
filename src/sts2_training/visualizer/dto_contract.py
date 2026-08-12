from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.visualizer.presentation import present_event as _present_event


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _frame_dto(record: Mapping[str, Any], frame_source: str) -> Mapping[str, Any]:
    """Return the exact DTO selected by the generic presenter for this event."""
    if frame_source == "final_dto":
        return _mapping(record.get("final_dto"))
    if frame_source == "received.masked_emulator_dto":
        return _mapping(_mapping(record.get("received")).get("masked_emulator_dto"))
    if frame_source == "result.masked_emulator_dto":
        return _mapping(_mapping(record.get("result")).get("masked_emulator_dto"))
    return {}


def _nonempty_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _intent_text(intent: Any, fallback: Any) -> Any:
    """Format the raw Emulator intent contract without guessing missing fields.

    GameInstance.BuildIntentDict exposes ``stateId`` as the pending move identifier and,
    for attack intents, ``attackDamage`` plus ``attackRepeats``. Those are already the
    engine-computed values, so the visualizer should display them directly rather than
    trying to infer attack damage from a move name or power state.
    """
    if not isinstance(intent, Mapping):
        return fallback

    state_id = intent.get("stateId")
    if state_id is None:
        return fallback

    move = str(state_id)
    attack_damage = intent.get("attackDamage")
    if attack_damage is None:
        return move

    attack = f"ATK {attack_damage}"
    repeats = intent.get("attackRepeats")
    if (
        isinstance(repeats, (int, float))
        and not isinstance(repeats, bool)
        and repeats > 1
    ):
        attack += f"×{repeats:g}" if isinstance(repeats, float) else f"×{repeats}"
    return f"{move} · {attack}"


def _apply_formal_emulator_fields(event: dict[str, Any], dto: Mapping[str, Any]) -> None:
    """Make the current raw Emulator DTO authoritative over legacy alias handling."""
    frame = event.get("frame")
    if not isinstance(frame, dict):
        return

    player = frame.get("player")
    if isinstance(player, dict):
        if dto.get("hp") is not None:
            player["current_hp"] = dto["hp"]
        if dto.get("maxHp") is not None:
            player["max_hp"] = dto["maxHp"]
        if dto.get("block") is not None:
            player["block"] = dto["block"]

    resources = frame.get("resources")
    if isinstance(resources, dict) and dto.get("totalFloor") is not None:
        # Emulator exposes both actFloor and totalFloor. The visualizer's single
        # FLOOR value is the run-wide floor number, matching GetRunSummary.FloorReached.
        resources["floor"] = dto["totalFloor"]

    raw_enemies = _sequence(dto.get("enemies"))
    shown_enemies = frame.get("enemies")
    if not isinstance(shown_enemies, list):
        return

    for shown, raw in zip(shown_enemies, raw_enemies):
        if not isinstance(shown, dict) or not isinstance(raw, Mapping):
            continue

        # Raw Emulator DTO: name is the localized LogName and can be empty in
        # headless TestMode. In that case id is the real monster model id; never
        # replace an available DTO identity with a synthetic "Enemy N" label.
        proper_name = _nonempty_string(raw.get("name"))
        monster_id = _nonempty_string(raw.get("id"))
        if proper_name is not None:
            shown["name"] = proper_name
        elif monster_id is not None:
            shown["name"] = monster_id

        if raw.get("hp") is not None:
            shown["current_hp"] = raw["hp"]
        if raw.get("maxHp") is not None:
            shown["max_hp"] = raw["maxHp"]
        if raw.get("block") is not None:
            shown["block"] = raw["block"]
        shown["intent"] = _intent_text(raw.get("intent"), shown.get("intent"))


def present_event(index: int, record: Mapping[str, Any]) -> dict[str, Any]:
    """Present an event with the formal current Emulator DTO fields taking priority."""
    event = _present_event(index, record)
    dto = _frame_dto(record, str(event.get("frame_source", "")))
    _apply_formal_emulator_fields(event, dto)
    return event


__all__ = ["present_event"]
