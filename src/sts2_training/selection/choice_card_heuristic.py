"""Shared heuristic preference for canonical ``choice_card`` decisions.

Only the versioned semantics/identity contract consumed by ``choice_semantics`` may
activate these preferences. Missing, malformed, future, or internally inconsistent
choice metadata stays neutral; this module never reconstructs mechanic meaning from
labels, card IDs, prompt text, selector names, or legacy operation fields.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.selection.action_classification import CHOICE_CARD_ACTION_TYPE, JsonObject
from sts2_training.selection.choice_semantics import choice_option_id, pending_choice_context

_PREFER_HIGH_QUALITY = frozenset({"gain", "upgrade", "retrieve", "play", "replay"})
_PREFER_LOW_QUALITY = frozenset({"discard", "exhaust", "remove", "transform"})

_CARD_TYPE_SCORE: dict[str, float] = {
    "Attack": 8.0,
    "Skill": 7.0,
    "Power": 6.0,
    "Curse": -50.0,
    "Status": -35.0,
}

_RARITY_SCORE: dict[str, float] = {
    "Rare": 3.0,
    "Uncommon": 1.5,
    "Common": 0.5,
}


def choice_card_preference_scores(
    legal_actions: Sequence[JsonObject],
    masked_emulator_dto: Mapping[str, Any],
) -> dict[str, float]:
    """Return per-action preference scores when the full canonical choice is usable.

    The result is all-or-nothing for one pending choice. If public selected/remaining
    identity is internally inconsistent, any available ``choice_card`` action cannot be
    matched to exactly one option, or action/option IDs are duplicated, an empty dict is
    returned so callers preserve neutral behavior.
    """

    choice_actions = [
        action for action in legal_actions if action.get("action_type") == CHOICE_CARD_ACTION_TYPE
    ]
    if not choice_actions:
        return {}

    context = pending_choice_context(masked_emulator_dto)
    if context is None or not context.semantics.is_known or not context.identity_valid:
        return {}

    operation = context.semantics.operation
    if operation in _PREFER_HIGH_QUALITY:
        direction = 1.0
    elif operation in _PREFER_LOW_QUALITY:
        direction = -1.0
    else:
        return {}

    pending = masked_emulator_dto.get("pendingChoice")
    if not isinstance(pending, Mapping):
        return {}
    raw_options = pending.get("options")
    if not isinstance(raw_options, Sequence) or isinstance(raw_options, (str, bytes)):
        return {}

    options_by_id: dict[str, Mapping[str, Any]] = {}
    for option in raw_options:
        if not isinstance(option, Mapping):
            return {}
        option_id = option.get("optionId")
        if not isinstance(option_id, str) or option_id not in context.option_ids:
            return {}
        if option_id in options_by_id:
            return {}
        options_by_id[option_id] = option

    scores: dict[str, float] = {}
    action_ids: set[str] = set()
    action_option_ids: set[str] = set()
    for action in choice_actions:
        action_id = action.get("action_id")
        option_id = choice_option_id(action)
        if not isinstance(action_id, str) or not action_id or option_id is None:
            return {}
        if action_id in action_ids or option_id in action_option_ids:
            return {}
        action_ids.add(action_id)
        action_option_ids.add(option_id)
        option = options_by_id.get(option_id)
        if option is None:
            return {}
        scores[action_id] = direction * card_quality_score(option)

    return scores


def card_quality_score(card: Mapping[str, Any]) -> float:
    """Small metadata-only card quality prior shared by canonical choice ranking."""

    card_type = card.get("type")
    rarity = card.get("rarity")
    score = _CARD_TYPE_SCORE.get(card_type, 0.0) + _RARITY_SCORE.get(rarity, 0.0)
    if card.get("upgraded") is True:
        score += 2.0
    upgrade_level = _finite_number(card.get("upgradeLevel"))
    if upgrade_level is not None and upgrade_level > 1:
        score += min(2.0, 0.5 * (upgrade_level - 1.0))
    return score


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None
