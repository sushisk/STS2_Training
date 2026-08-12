"""Decision logic for the initial data-collection stage.

Picks uniformly at random within a priority-ordered `action_type` category, with two
scored exceptions: `choice_card` decisions use the canonical-semantics card-quality
preference, and `card` (which card to play) decisions use an epsilon-greedy wrapper
around `PriorHeuristicPolicy` - the same action-quality prior Beam search ranks nodes
with - instead of a uniform-random pick, so card play stops being pure noise while
still retaining `epsilon` exploration for data diversity. See `how_to_use.md` for the
expected input shape and where this plugs into the API client.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from sts2_training.selection.action_classification import (
    CARD_ACTION_TYPE,
    CHOICE_CARD_ACTION_TYPE,
    CHOICE_CONFIRM_ACTION_TYPE,
    CHOICE_REWARD_POTION_REPLACE_ACTION_TYPE,
    CHOICE_REWARD_POTION_TAKE_ACTION_TYPE,
    CHOICE_REWARD_SKIP_ACTION_TYPE,
    CHOICE_SKIP_ACTION_TYPE,
    MAP_ROOM_ACTION_TYPE,
    JsonObject,
    available_actions,
    group_by_action_type,
)
from sts2_training.selection.choice_card_heuristic import choice_card_preference_scores
from sts2_training.selection.room_heuristic import room_preference_scores

if TYPE_CHECKING:
    # Deferred to a lazy import in HeuristicCombatSelector.__init__ - decision.policy's
    # package sits behind decision/__init__.py, which imports this module's own
    # CombatDecisionEngine dependency, so a module-level import here would be circular.
    from sts2_training.decision.policy import PolicyModel

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
    random. `card` decisions are epsilon-greedy over `PriorHeuristicPolicy`'s score (see
    `_choose_card`). `map_room` (Whole Run map navigation) decisions are epsilon-greedy
    over `room_heuristic.room_preference_scores` (see `_choose_room`), checked before
    `_CATEGORY_PRIORITY` since it never co-occurs with combat/choice categories - both
    live at mutually exclusive decision boundaries. Reward decisions receive one
    conservative potion-specific rule: take a potion when an empty-slot TAKE is
    explicitly published, otherwise prefer skipping a full-belt PotionReward over
    randomly discarding an existing potion; if skipping is disallowed, choose among the
    available replacement slots. Other unknown action types continue to fall back to a
    single random pool.
    """

    def __init__(
        self,
        rng: random.Random | None = None,
        *,
        epsilon: float = 0.1,
        policy: "PolicyModel | None" = None,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0.0 and 1.0")
        self._rng = rng or random.Random()
        self._epsilon = epsilon
        if policy is not None:
            self._policy = policy
        else:
            from sts2_training.decision.policy import PriorHeuristicPolicy

            self._policy = PriorHeuristicPolicy()

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

        map_rooms = by_type.get(MAP_ROOM_ACTION_TYPE)
        if map_rooms:
            return self._choose_room(map_rooms, dto)

        for action_type in _CATEGORY_PRIORITY:
            candidates = by_type.get(action_type)
            if not candidates:
                continue
            if action_type == CHOICE_CARD_ACTION_TYPE:
                return self._choose_choice_card(candidates, dto)
            if action_type == CARD_ACTION_TYPE:
                return self._choose_card(candidates, dto)
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

    def _choose_card(
        self,
        candidates: Sequence[JsonObject],
        masked_emulator_dto: Mapping[str, Any],
    ) -> JsonObject:
        """Epsilon-greedy over `PriorHeuristicPolicy`'s score - `epsilon` of the time
        (data-diversity exploration, matching this class's random baseline elsewhere),
        picks uniformly at random; otherwise plays `_policy`'s top-ranked card instead
        of ignoring card quality/targeting/danger entirely."""
        if self._rng.random() < self._epsilon:
            return self._choose(candidates)

        proposals = self._policy.propose(candidates, masked_emulator_dto, top_k=1)
        if not proposals:
            return self._choose(candidates)
        best_action_id = proposals[0].action_id
        best_match = next(
            (action for action in candidates if action.get("action_id") == best_action_id),
            None,
        )
        return best_match if best_match is not None else self._choose(candidates)

    def _choose_room(
        self,
        candidates: Sequence[JsonObject],
        masked_emulator_dto: Mapping[str, Any],
    ) -> JsonObject:
        """Epsilon-greedy over `room_heuristic.room_preference_scores` - same
        exploration/exploit split as `_choose_card`, see its docstring."""
        if self._rng.random() < self._epsilon:
            return self._choose(candidates)

        scores = room_preference_scores(candidates, masked_emulator_dto)
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
