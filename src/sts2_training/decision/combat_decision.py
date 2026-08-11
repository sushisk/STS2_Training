"""Combat decision-phase ontology shared by Beam/search-mode integration.

Search modes describe latency/quality knobs. This module describes semantic scope: which
live action types belong to a stable combat decision versus an interactive continuation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

STABLE_COMBAT_ACTION_TYPES = frozenset({"system", "card", "potion"})
CONTINUATION_ACTION_TYPES = frozenset(
    {"choice_target", "choice_card", "choice_confirm", "choice_skip"}
)
COMPLETION_ACTION_TYPES = frozenset({"choice_confirm", "choice_skip"})
COMBAT_BEAM_ACTION_TYPES = STABLE_COMBAT_ACTION_TYPES | CONTINUATION_ACTION_TYPES


def available_action_types(dto: Mapping[str, Any]) -> set[str] | None:
    legal_actions = dto.get("legal_actions")
    if not isinstance(legal_actions, Sequence) or isinstance(legal_actions, (str, bytes)):
        return None
    action_types: set[str] = set()
    for action in legal_actions:
        if not isinstance(action, Mapping):
            return None
        if action.get("is_available") is False:
            continue
        action_type = action.get("action_type")
        if not isinstance(action_type, str):
            return None
        action_types.add(action_type)
    return action_types


def is_continuation_decision(dto: Mapping[str, Any]) -> bool:
    action_types = available_action_types(dto)
    return (
        action_types is not None
        and bool(action_types)
        and action_types <= CONTINUATION_ACTION_TYPES
    )


def action_type_for_id(dto: Mapping[str, Any], action_id: str) -> str | None:
    legal_actions = dto.get("legal_actions")
    if not isinstance(legal_actions, Sequence) or isinstance(legal_actions, (str, bytes)):
        return None
    for action in legal_actions:
        if not isinstance(action, Mapping) or action.get("is_available") is False:
            continue
        if action.get("action_id") != action_id:
            continue
        action_type = action.get("action_type")
        return action_type if isinstance(action_type, str) else None
    return None
