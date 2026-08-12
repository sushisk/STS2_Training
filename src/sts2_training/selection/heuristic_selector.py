"""Heuristic action selection for the initial data-collection stage."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from sts2_training.selection.action_classification import (
    CARD_ACTION_TYPE,
    CHOICE_CARD_ACTION_TYPE,
    CHOICE_CONFIRM_ACTION_TYPE,
    CHOICE_EVENT_OPTION_ACTION_TYPE,
    CHOICE_REWARD_CARD_ACTION_TYPE,
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
from sts2_training.selection.event_choice_heuristic import safe_event_option_candidates
from sts2_training.selection.reward_card_selection import (
    RandomRewardCardSelectionPolicy,
    RewardCardSelectionPolicy,
)
from sts2_training.selection.room_heuristic import room_preference_scores

if TYPE_CHECKING:
    # Keep this import type-only to avoid the decision package's circular import path.
    from sts2_training.decision.policy import PolicyModel

_CATEGORY_PRIORITY = (
    CARD_ACTION_TYPE,
    CHOICE_CARD_ACTION_TYPE,
    CHOICE_CONFIRM_ACTION_TYPE,
    CHOICE_SKIP_ACTION_TYPE,
)


class NoAvailableActionError(RuntimeError):
    """Raised when a decision has no action Training is willing to select."""

    def __init__(
        self,
        message: str,
        *,
        decision: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.decision = dict(decision) if decision is not None else None


class HeuristicCombatSelector:
    """Choose an available action using category-specific heuristics where defined."""

    def __init__(
        self,
        rng: random.Random | None = None,
        *,
        epsilon: float = 0.1,
        policy: PolicyModel | None = None,
        reward_card_policy: RewardCardSelectionPolicy | None = None,
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
        self._reward_card_policy = (
            reward_card_policy
            if reward_card_policy is not None
            else RandomRewardCardSelectionPolicy()
        )

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

        event_options = by_type.get(CHOICE_EVENT_OPTION_ACTION_TYPE)
        if event_options:
            return self._choose_event_option(event_options)

        for action_type in _CATEGORY_PRIORITY:
            candidates = by_type.get(action_type)
            if not candidates:
                continue
            if action_type == CHOICE_CARD_ACTION_TYPE:
                return self._choose_choice_card(candidates, dto)
            if action_type == CARD_ACTION_TYPE:
                return self._choose_card(candidates, dto)
            return self._choose(candidates)

        reward_cards = by_type.get(CHOICE_REWARD_CARD_ACTION_TYPE)
        if reward_cards:
            # Reward-card decisions are a distinct Emulator boundary from choice_card.
            # Delegate the whole available action set so a future policy can compare
            # cards against skip/reroll/alternative actions in one place.
            return self._reward_card_policy.select(actions, dto, rng=self._rng)

        return self._choose(actions)

    def _choose_choice_card(
        self,
        candidates: Sequence[JsonObject],
        masked_emulator_dto: Mapping[str, Any],
    ) -> JsonObject:
        scores = choice_card_preference_scores(candidates, masked_emulator_dto)
        return self._choose_best_scored(candidates, scores)

    def _choose_card(
        self,
        candidates: Sequence[JsonObject],
        masked_emulator_dto: Mapping[str, Any],
    ) -> JsonObject:
        """Use epsilon-greedy selection over the policy's top proposal."""
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
        """Use epsilon-greedy selection over room preference scores."""
        if self._rng.random() < self._epsilon:
            return self._choose(candidates)

        scores = room_preference_scores(candidates, masked_emulator_dto)
        return self._choose_best_scored(candidates, scores)

    def _choose_event_option(self, candidates: Sequence[JsonObject]) -> JsonObject:
        """Never voluntarily choose a confirmed-lethal option while a safer one exists -
        a hard safety constraint, not subject to `epsilon` exploration (unlike card/room
        selection's soft quality preferences). See `event_choice_heuristic`'s docstring:
        beyond lethality there is no generic quality signal for event options, so the
        (already lethality-filtered) candidates are chosen from uniformly at random."""
        return self._choose(safe_event_option_candidates(candidates))

    def _choose_best_scored(
        self,
        candidates: Sequence[JsonObject],
        scores: Mapping[str, float],
    ) -> JsonObject:
        if not scores:
            return self._choose(candidates)

        best_score = max(scores.values())
        preferred = [
            action for action in candidates if scores.get(action.get("action_id")) == best_score
        ]
        return self._choose(preferred or candidates)

    def _choose(self, candidates: Sequence[JsonObject]) -> JsonObject:
        return candidates[self._rng.randrange(len(candidates))]
