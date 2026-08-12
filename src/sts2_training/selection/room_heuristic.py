"""Shared heuristic preference for ``map_room`` (Whole Run map navigation) decisions.

Deliberately separated from `choice_card_heuristic.py`/`heuristic_selector.py`'s combat
scope, mirroring that module's structure: a pure `legal_actions + masked_emulator_dto ->
per-action_id score` function, with no side effects and no dependency on any particular
caller. This is a first pass - `_POINT_TYPE_SCORE` and the low-HP RestSite/Elite
adjustments are an initial, easily-tunable baseline, not a final strategy.

Missing/malformed room data (no ``point_type``, non-numeric hp/maxHp, etc.) degrades
that single candidate/run to neutral rather than raising, matching this package's other
heuristics' "stay neutral on uncertainty" convention.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.selection.action_classification import MAP_ROOM_ACTION_TYPE, JsonObject

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
    """Per-action_id score for every available ``map_room`` candidate, highest-first.

    Returns an empty dict (neutral - callers should fall back to their own default,
    e.g. uniform random) when there are no ``map_room`` candidates at all.
    """

    room_actions = [
        action for action in legal_actions if action.get("action_type") == MAP_ROOM_ACTION_TYPE
    ]
    if not room_actions:
        return {}

    hp_ratio = _hp_ratio(masked_emulator_dto)

    scores: dict[str, float] = {}
    for action in room_actions:
        action_id = action.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            continue
        scores[action_id] = room_quality_score(action, hp_ratio)
    return scores


def room_quality_score(room_action: Mapping[str, Any], hp_ratio: "float | None") -> float:
    """Small metadata-only room quality prior for one ``map_room`` candidate."""

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


def _hp_ratio(masked_emulator_dto: Mapping[str, Any]) -> "float | None":
    hp = _finite_number(masked_emulator_dto.get("hp"))
    max_hp = _finite_number(masked_emulator_dto.get("maxHp"))
    if hp is None or max_hp is None or max_hp <= 0:
        return None
    return max(0.0, hp) / max_hp


def _finite_number(value: Any) -> "float | None":
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None
