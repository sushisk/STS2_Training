"""Placeholder decision logic for the initial data-collection stage.

Picks uniformly at random within a priority-ordered `action_type` category, so that
combat decisions can be driven end-to-end (and logged) before any learned or
hand-tuned per-category logic exists. See `how_to_use.md` for the expected input
shape and where this plugs into the API client.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.selection.action_classification import (
    CARD_ACTION_TYPE,
    CHOICE_CARD_ACTION_TYPE,
    CHOICE_CONFIRM_ACTION_TYPE,
    CHOICE_SKIP_ACTION_TYPE,
    JsonObject,
    available_actions,
    group_by_action_type,
)

_CATEGORY_PRIORITY = (
    CARD_ACTION_TYPE,
    CHOICE_CARD_ACTION_TYPE,
    CHOICE_CONFIRM_ACTION_TYPE,
    CHOICE_SKIP_ACTION_TYPE,
)


class NoAvailableActionError(RuntimeError):
    """Raised when a decision has no action Training is willing to select.

    `decision` is optional context for callers that need to distinguish a genuine
    terminal decision from a malformed/non-terminal decision with no selectable action.
    """

    def __init__(
        self,
        message: str,
        *,
        decision: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.decision = dict(decision) if decision is not None else None


class HeuristicCombatSelector:
    """Classifies `legal_actions` by `action_type` and picks one at random.

    Categories are tried in `_CATEGORY_PRIORITY` order; the first non-empty category
    is chosen from. Any `action_type` outside that list (e.g. `potion`, `end_turn`)
    falls back to a single "other" pool. Replace `_choose` (or split it per category)
    to swap in real heuristics later without touching the classification code.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def select(self, legal_actions: Sequence[JsonObject]) -> JsonObject:
        actions = available_actions(legal_actions)
        if not actions:
            raise NoAvailableActionError("no available legal_actions to select from")

        by_type = group_by_action_type(legal_actions)
        for action_type in _CATEGORY_PRIORITY:
            candidates = by_type.get(action_type)
            if candidates:
                return self._choose(candidates)

        return self._choose(actions)

    def _choose(self, candidates: Sequence[JsonObject]) -> JsonObject:
        return candidates[self._rng.randrange(len(candidates))]
