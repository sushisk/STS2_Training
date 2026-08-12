"""Heuristic preference scores for ``map_room`` navigation decisions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.selection.action_classification import JsonObject, map_room_actions

_POINT_TYPE_SCORE: dict[str, float] = {
    "Treasure": 6.0,
    "RestSite": 5.0,
    "Shop": 3.0,
    "Monster": 0.0,
    "Unknown": -1.0,
    "Elite": -2.0,
    "Boss": 0.0,
}

_LOW_HP_RATIO_THRESHOLD = 0.5


def room_preference_scores(
    legal_actions: Sequence[JsonObject],
    masked_emulator_dto: Mapping[str, Any],
) -> dict[str, float]:
    """Return per-action preference scores for available ``map_room`` candidates."""
    room_actions = map_room_actions(legal_actions)
    if not room_actions:
        return {}

    hp_ratio = _hp_ratio(masked_emulator_dto)
    scores: dict[str, float] = {}
    for action in room_actions:
        action_id = action.get("action_id")
        if isinstance(action_id, str) and action_id:
            scores[action_id] = room_quality_score(action, hp_ratio)
    return scores


def room_quality_score(room_action: Mapping[str, Any], hp_ratio: float | None) -> float:
    """Return a metadata-only quality prior for one room candidate."""
    params = room_action.get("parameters")
    point_type = params.get("point_type") if isinstance(params, Mapping) else None
    score = _POINT_TYPE_SCORE.get(point_type, 0.0) if isinstance(point_type, str) else 0.0

    if hp_ratio is not None and hp_ratio < _LOW_HP_RATIO_THRESHOLD:
        urgency = (_LOW_HP_RATIO_THRESHOLD - hp_ratio) / _LOW_HP_RATIO_THRESHOLD
        if point_type == "RestSite":
            score += 10.0 * urgency
        elif point_type == "Elite":
            score -= 8.0 * urgency

    return score


def _hp_ratio(masked_emulator_dto: Mapping[str, Any]) -> float | None:
    hp = _finite_number(masked_emulator_dto.get("hp"))
    max_hp = _finite_number(masked_emulator_dto.get("maxHp"))
    if hp is None or max_hp is None or max_hp <= 0:
        return None
    return max(0.0, hp) / max_hp


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None
