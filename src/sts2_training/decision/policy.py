"""`PolicyModel`: proposes a ranked shortlist of candidate actions per decision.

`BeamSearchEngine` calls `propose_batch` once per beam depth, covering every
frontier node in that depth in a single call - a real learned policy should
override `propose_batch` directly (one batched forward pass over every node)
to hit the ~5ms/batch latency budget this is designed around; the default
`propose_batch` here just loops over `propose`.

`PriorHeuristicPolicy` is the no-model default: the same `action_type`
category-priority idea as `selection.heuristic_selector.HeuristicCombatSelector`,
but ranking a list of top-k candidates instead of picking one, since beam
search needs multiple candidate actions per node to actually branch on. It
exists so `BeamSearchEngine`/`CombatDecisionEngine` are runnable and testable
before any trained policy checkpoint exists - swap in a real model by
implementing `PolicyModel`.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sts2_training.selection.action_classification import (
    CARD_ACTION_TYPE,
    CHOICE_CARD_ACTION_TYPE,
    CHOICE_CONFIRM_ACTION_TYPE,
    CHOICE_SKIP_ACTION_TYPE,
    available_actions,
    group_by_action_type,
)

JsonObject = Mapping[str, Any]

_CATEGORY_PRIORITY = (
    CARD_ACTION_TYPE,
    CHOICE_CARD_ACTION_TYPE,
    CHOICE_CONFIRM_ACTION_TYPE,
    CHOICE_SKIP_ACTION_TYPE,
)


@dataclass(frozen=True)
class ActionCandidate:
    """One action a `PolicyModel` proposes for beam search to branch on."""

    action_id: str


class PolicyModel(ABC):
    """Proposes candidate actions for one decision (`legal_actions` plus the
    full `masked_emulator_dto` they came from), best-first, capped at `top_k`.
    """

    @abstractmethod
    def propose(
        self,
        legal_actions: Sequence[JsonObject],
        masked_emulator_dto: Mapping[str, Any],
        *,
        top_k: int,
    ) -> list[ActionCandidate]:
        raise NotImplementedError

    def propose_batch(
        self,
        requests: Sequence[tuple[Sequence[JsonObject], Mapping[str, Any]]],
        *,
        top_k: int,
    ) -> list[list[ActionCandidate]]:
        """Batched counterpart of `propose`, one entry per request, in order.

        Override this directly in a learned policy for real batched inference;
        the default here is a plain loop and carries none of the throughput
        benefit the batched call is meant to provide.
        """
        return [
            self.propose(legal_actions, dto, top_k=top_k) for legal_actions, dto in requests
        ]


class PriorHeuristicPolicy(PolicyModel):
    """No-model default. Ranks available actions by the same category
    priority as `HeuristicCombatSelector` (card > choice_card > choice_confirm
    > choice_skip > everything else - e.g. potion/end_turn), preserving
    `legal_actions` order within each category. Deterministic unless `rng` is
    given, in which case each category is shuffled before ranking - useful for
    exploration diversity when `top_k` truncates a category (e.g. many
    playable hand cards).
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng

    def propose(
        self,
        legal_actions: Sequence[JsonObject],
        masked_emulator_dto: Mapping[str, Any],
        *,
        top_k: int,
    ) -> list[ActionCandidate]:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        actions = available_actions(legal_actions)
        if not actions:
            return []

        by_type = group_by_action_type(legal_actions)
        present_types = {a.get("action_type") for a in actions}
        other_types = sorted(t for t in present_types - set(_CATEGORY_PRIORITY) if t is not None)

        ordered: list[JsonObject] = []
        for action_type in (*_CATEGORY_PRIORITY, *other_types):
            candidates = list(by_type.get(action_type, ()))
            if self._rng is not None:
                self._rng.shuffle(candidates)
            ordered.extend(candidates)

        return [ActionCandidate(action_id=action["action_id"]) for action in ordered[:top_k]]
