from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _path(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _pick(value: Mapping[str, Any], paths: Sequence[str], default: Any = None) -> Any:
    for path in paths:
        candidate = _path(value, path)
        if candidate is not None:
            return candidate
    return default


def _masked_dto(container: Any) -> dict[str, Any]:
    value = _mapping(container).get("masked_emulator_dto")
    return dict(value) if isinstance(value, Mapping) else {}


def _status_label(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _pick(value, ("name", "display_name", "id", "type"), deepcopy(dict(value)))
    return deepcopy(value)


def _entity_view(value: Any, *, fallback_name: str) -> dict[str, Any]:
    entity = _mapping(value)
    current_hp = _pick(entity, ("current_hp", "hp", "health", "current_health"))
    return {
        "name": _pick(
            entity,
            ("name", "display_name", "id", "character_id", "monster_id"),
            fallback_name,
        ),
        "current_hp": current_hp,
        "max_hp": _pick(entity, ("max_hp", "max_health", "health_max"), current_hp),
        "block": _pick(entity, ("block", "current_block"), 0),
        "intent": _pick(entity, ("intent", "next_move", "move_intent", "intent_damage")),
        "statuses": [
            _status_label(status)
            for status in _sequence(
                _pick(entity, ("statuses", "powers", "effects", "buffs"), [])
            )
        ],
    }


def _action_view(value: Any) -> dict[str, Any]:
    action = _mapping(value)
    action_id = _pick(action, ("action_id", "id", "card_id", "uuid"), "")
    return {
        "action_id": action_id,
        "action_type": _pick(action, ("action_type", "type")),
        "name": _pick(
            action,
            ("name", "display_name", "card_name", "action_type"),
            action_id,
        ),
        "cost": _pick(
            action,
            ("cost", "energy_cost", "parameters.cost", "parameters.energy"),
            "",
        ),
        "description": _pick(
            action,
            ("description", "text", "rules_text", "effect", "parameters"),
            "",
        ),
        "is_available": _pick(action, ("is_available", "playable", "enabled"), True) is not False,
        "parameters": deepcopy(_mapping(action.get("parameters"))),
    }


def _frame_view(dto: Mapping[str, Any]) -> dict[str, Any]:
    player_raw = _mapping(
        _pick(
            dto,
            ("player", "player_state", "combat.player", "combat_state.player", "state.player"),
            {},
        )
    )
    enemies_raw = _sequence(
        _pick(
            dto,
            ("enemies", "monsters", "combat.enemies", "combat_state.enemies", "state.enemies"),
            [],
        )
    )
    hand_raw = _sequence(
        _pick(
            dto,
            ("hand", "cards_in_hand", "combat.hand", "combat_state.hand", "player.hand", "state.hand"),
            [],
        )
    )
    legal_raw = _sequence(dto.get("legal_actions"))
    if not hand_raw:
        card_actions = [
            action
            for action in legal_raw
            if any(
                token in str(_mapping(action).get("action_type", "")).lower()
                for token in ("card", "play_card", "card_play")
            )
        ]
        hand_raw = card_actions or legal_raw[:9]

    def player_or_dto(paths: Sequence[str], default: Any = None) -> Any:
        value = _pick(player_raw, paths)
        return _pick(dto, paths, default) if value is None else value

    player = _entity_view(player_raw, fallback_name="Player")
    player["name"] = player_or_dto(
        ("character_id", "character", "name"), player["name"]
    )
    if player["current_hp"] is None:
        player["current_hp"] = player_or_dto(("current_hp", "hp", "health"))
    if player["max_hp"] is None:
        player["max_hp"] = player_or_dto(("max_hp", "max_health"), player["current_hp"])

    return {
        "boundary": _pick(dto, ("boundary", "room_context.type", "room.type")),
        "outcome": _pick(dto, ("outcome", "run_result", "run_outcome")),
        "player": player,
        "enemies": [
            _entity_view(enemy, fallback_name=f"Enemy {index + 1}")
            for index, enemy in enumerate(enemies_raw[:5])
        ],
        "hand": [_action_view(action) for action in hand_raw[:12]],
        "resources": {
            "character": player_or_dto(("character_id", "character", "name")),
            "gold": player_or_dto(("gold", "money", "coins")),
            "floor": _pick(dto, ("floor", "floor_number", "current_floor", "map.floor")),
            "energy": player_or_dto(("energy", "current_energy")),
            "max_energy": player_or_dto(("max_energy", "energy_max")),
        },
        "piles": {
            "draw": _pick(dto, ("draw_pile_count", "draw_count", "piles.draw", "deck.draw_count")),
            "discard": _pick(dto, ("discard_pile_count", "discard_count", "piles.discard", "deck.discard_count")),
            "exhaust": _pick(dto, ("exhaust_pile_count", "exhaust_count", "piles.exhaust")),
            "deck": _pick(dto, ("deck_size", "deck.count", "deck_count")),
        },
    }


def _selected_action(record: Mapping[str, Any], selection_dto: Mapping[str, Any]) -> dict[str, Any] | None:
    selected_id = record.get("selected_action_id")
    if not isinstance(selected_id, str) or not selected_id:
        request_id = _mapping(record.get("request")).get("action_id")
        selected_id = request_id if isinstance(request_id, str) else None
    if not selected_id:
        return None

    legal_actions = _sequence(selection_dto.get("legal_actions"))
    for action in legal_actions:
        if isinstance(action, Mapping) and action.get("action_id") == selected_id:
            return _action_view(action)
    return _action_view({"action_id": selected_id})


def _frame_input(record: Mapping[str, Any]) -> tuple[dict[str, Any], str, str]:
    final_dto = record.get("final_dto")
    if isinstance(final_dto, Mapping):
        return deepcopy(dict(final_dto)), "final_dto", "run_terminal"

    received = _masked_dto(record.get("received"))
    if received:
        operation = _mapping(record.get("request")).get("operation")
        phase = "beam_explore" if operation == "emulate_actions" else "selection"
        return received, "received.masked_emulator_dto", phase

    result = _masked_dto(record.get("result"))
    if result:
        return result, "result.masked_emulator_dto", "result"
    return {}, "none", "unknown"


def present_event(index: int, record: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one JSONL record into the sole stable browser-facing contract.

    DTO aliases and record-shape differences are resolved here. The browser consumes
    only ``frame`` plus event metadata; the original record remains available under
    ``raw`` for the inspector.
    """
    received = _mapping(record.get("received"))
    result = _mapping(record.get("result"))
    request = _mapping(record.get("request"))
    selection_dto = _masked_dto(received)
    frame_dto, frame_source, phase = _frame_input(record)
    operation = request.get("operation")
    branch_id = request.get("branch_id") or received.get("branch_id") or result.get("branch_id")
    return {
        "index": index,
        "event": record.get("event", "unknown"),
        "logged_at": record.get("logged_at"),
        "operation": operation,
        "branch_id": branch_id,
        "decision_point_id": received.get("decision_point_id") or result.get("decision_point_id"),
        "phase": phase,
        "frame_source": frame_source,
        "frame": _frame_view(frame_dto),
        "selected_action_id": record.get("selected_action_id") or request.get("action_id"),
        "selected_action": _selected_action(record, selection_dto),
        "client_error": deepcopy(record.get("client_error")),
        "room_result": deepcopy(record.get("room_result")),
        "run_result": deepcopy(record.get("run_result")),
        "raw": deepcopy(dict(record)),
    }


__all__ = ["present_event"]
