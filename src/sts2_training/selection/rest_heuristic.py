"""Heuristic preference scores for ``choice_rest_option`` (campfire) decisions.

A rest site trades one permanent improvement against HP. The Emulator heals
``maxHp * 0.3`` (`Core.Entities.RestSite/HealRestSiteOption.GetBaseHealAmount`), and
``MEND`` heals the same amount, so the value of resting is proportional to how much HP is
actually missing while every other option's value is roughly constant.

Until this module existed, `HeuristicCombatSelector` had no branch for
``choice_rest_option`` and fell through to a uniform random pick. In a 5-run floor-reach
evaluation that produced SMITH 7 / HEAL 4, including SMITH at 8 HP, 18 HP and 30 HP - the
runs had no other source of healing and both deepest runs entered the Act 1 boss at 8 and
4 HP.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.selection.action_classification import (
    JsonObject,
    rest_option_actions,
)

# Options that convert missing HP into HP. Anything else is a permanent upgrade whose
# value does not depend on current HP.
HEALING_REST_OPTION_IDS = frozenset({"HEAL", "MEND"})

# A heal is worth the HP it restores, so score it by the fraction of maxHp missing.
_HEAL_SCORE_PER_MISSING_HP_RATIO = 10.0

# SMITH (upgrade a card) is the always-available alternative. 3.0 puts the crossover at
# 30% missing HP: heal below 70% HP, upgrade at or above it. Below 70% a full rest is
# never wasted (the heal is 30% of maxHp), and above it part of the heal would be.
_SMITH_SCORE = 3.0

_SMITH_REST_OPTION_ID = "SMITH"


def rest_option_preference_scores(
    legal_actions: Sequence[JsonObject],
    masked_emulator_dto: Mapping[str, Any],
) -> dict[str, float]:
    """Return per-action preference scores for available ``choice_rest_option`` actions."""

    options = rest_option_actions(legal_actions)
    if not options:
        return {}

    hp_ratio = _hp_ratio(masked_emulator_dto)
    scores: dict[str, float] = {}
    for action in options:
        action_id = action.get("action_id")
        if isinstance(action_id, str) and action_id:
            scores[action_id] = rest_option_quality_score(_option_id(action), hp_ratio)
    return scores


def rest_option_quality_score(option_id: str | None, hp_ratio: float | None) -> float:
    """Return a quality score for one rest option.

    Unrecognized options score 0.0 rather than guessing: this module knows what healing is
    worth and what SMITH is worth, and has no comparable signal for LIFT/DIG/CLONE/COOK/
    HATCH/KINDLE. They stay selectable, just never preferred over a needed heal.
    """

    if option_id in HEALING_REST_OPTION_IDS:
        if hp_ratio is None:
            # HP unreadable: healing is the choice that cannot be catastrophic.
            return _HEAL_SCORE_PER_MISSING_HP_RATIO
        missing = max(0.0, min(1.0, 1.0 - hp_ratio))
        return _HEAL_SCORE_PER_MISSING_HP_RATIO * missing
    if option_id == _SMITH_REST_OPTION_ID:
        return _SMITH_SCORE
    return 0.0


def _option_id(action: Mapping[str, Any]) -> str | None:
    params = action.get("parameters")
    option_id = params.get("restOptionId") if isinstance(params, Mapping) else None
    if isinstance(option_id, str) and option_id:
        return option_id
    label = action.get("label")
    return label if isinstance(label, str) and label else None


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
