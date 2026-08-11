"""Placeholder decision logic for the initial data-collection stage.

Picks uniformly at random within a priority-ordered `action_type` category, with one
canonical-semantics-aware rule for `choice_card` decisions. See `how_to_use.md` for the
expected input shape and where this plugs into the API client.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.selection.action_classification import (
    CARD_ACTION_TYPE,
    CHOICE_CARD_ACTION_TYPE,
    CHOICE_CONFIRM_ACTION_TYPE,
    CHOICE_REWARD_POTION_REPLACE_ACTION_TYPE,
    CHOICE_REWARD_POTION_TAKE_ACTION_TYPE,
    CHOICE_REWARD_SKIP_ACTION_TYPE,
    CHOICE_SKIP_ACTION_TYPE,
    JsonObject,
    available_actions,
    group_by_action_type,
)
from sts2_training.selection.choice_card_heuristic import choice_card_preference_scores

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
    """Classifies `legal_actions` by `action_type` and chooses one candidate.

    Categories are tried in `_CATEGORY_PRIORITY` order; the first non-empty category
    is chosen from. Canonical v1 `choice_card` semantics use the same card-quality
    preference as the Beam prior, while missing/malformed/future semantics remain
    random. Reward decisions receive one conservative potion-specific rule: take a
    potion when an empty-slot TAKE is explicitly published, otherwise prefer skipping a
    full-belt PotionReward over randomly discarding an existing potion; if skipping is
    disallowed, choose among the available replacement slots. Other unknown action types
    continue to fall back to a single random pool.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def select(
        self,
        legal_actions: Sequence[JsonObject],
        masked_emulator_dto: Mapping[str, Any] | None = None,
    ) -> JsonObject:
        actions = available_actions(legal_actions)
        if not actions:
            raise NoAvailableActionError("no available legal_actions to select from")

        by_type = group_by_action_type(actions)
        potion_takes = by_type.get(CHOICE_REWARD_POTION_TAKE_ACTION_TYPE)
        if potion_takes:
            return self._choose(potion_takes)

        potion_replacements = by_type.get(CHOICE_REWARD_POTION_REPLACE_ACTION_TYPE)
        if potion_replacements:
            reward_skips = by_type.get(CHOICE_REWARD_SKIP_ACTION_TYPE)
            if reward_skips:
                return self._choose(reward_skips)
            return self._choose(potion_replacements)

        dto = masked_emulator_dto if masked_emulator_dto is not None else {}
        for action_type in _CATEGORY_PRIORITY:
            candidates = by_type.get(action_type)
            if not candidates:
                continue
            if action_type == CHOICE_CARD_ACTION_TYPE:
                return self._choose_choice_card(candidates, dto)
            return self._choose(candidates)

        return self._choose(actions)

    def _choose_choice_card(
        self,
        candidates: Sequence[JsonObject],
        masked_emulator_dto: Mapping[str, Any],
    ) -> JsonObject:
        scores = choice_card_preference_scores(candidates, masked_emulator_dto)
        if not scores:
            return self._choose(candidates)

        best_score = max(scores.values())
        preferred = [
            action
            for action in candidates
            if scores.get(action.get("action_id")) == best_score
        ]
        return self._choose(preferred or candidates)

    def _choose(self, candidates: Sequence[JsonObject]) -> JsonObject:
        return candidates[self._rng.randrange(len(candidates))]
