"""Structural candidate coverage applied after policy ranking.

A PolicyModel owns action preference/ranking.  This module owns the small set of
search-topology invariants that should survive swapping the heuristic policy for a
learned one: retain a turn-boundary branch, retain at least one card branch in ordinary
combat, keep potion diversity when there is room, and retain a legal pending-choice
completion branch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sts2_training.decision.policy import ActionCandidate, PolicyModel
from sts2_training.selection.action_classification import (
    CARD_ACTION_TYPE,
    CHOICE_CARD_ACTION_TYPE,
    CHOICE_CONFIRM_ACTION_TYPE,
    CHOICE_SKIP_ACTION_TYPE,
)

JsonObject = Mapping[str, object]

_SYSTEM_ACTION_TYPE = "system"
_POTION_ACTION_TYPE = "potion"


class CoverageConstrainedPolicy(PolicyModel):
    """Policy adapter that applies structural branch coverage after ranking."""

    def __init__(self, inner: PolicyModel) -> None:
        self._inner = inner

    @property
    def inner(self) -> PolicyModel:
        return self._inner

    def propose(
        self,
        legal_actions: Sequence[JsonObject],
        masked_emulator_dto: Mapping[str, object],
        *,
        top_k: int,
    ) -> list[ActionCandidate]:
        ranked = self._inner.propose(
            legal_actions, masked_emulator_dto, top_k=top_k
        )
        return apply_structural_coverage(ranked, legal_actions, top_k=top_k)

    def propose_batch(
        self,
        requests: Sequence[
            tuple[Sequence[JsonObject], Mapping[str, object]]
        ],
        *,
        top_k: int,
    ) -> list[list[ActionCandidate]]:
        ranked = self._inner.propose_batch(requests, top_k=top_k)
        if len(ranked) != len(requests):
            # Keep the PolicyModel contract failure explicit instead of silently
            # misaligning coverage with a different request.
            raise RuntimeError(
                "PolicyModel.propose_batch must return exactly one entry per request"
            )
        return [
            apply_structural_coverage(candidates, legal_actions, top_k=top_k)
            for candidates, (legal_actions, _dto) in zip(ranked, requests)
        ]


def apply_structural_coverage(
    ranked: Sequence[ActionCandidate],
    legal_actions: Sequence[JsonObject],
    *,
    top_k: int,
) -> list[ActionCandidate]:
    """Retain structural branches without changing the policy's relative ranking."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    available = [
        action
        for action in legal_actions
        if action.get("is_available") is not False
        and isinstance(action.get("action_id"), str)
        and isinstance(action.get("action_type"), str)
    ]
    by_id = {str(action["action_id"]): action for action in available}

    selected: list[ActionCandidate] = []
    seen: set[str] = set()
    for candidate in ranked:
        if not isinstance(candidate, ActionCandidate):
            raise RuntimeError("PolicyModel must return ActionCandidate objects")
        if candidate.action_id not in by_id:
            raise RuntimeError(
                "PolicyModel proposed an action_id that is not currently available: "
                f"{candidate.action_id!r}"
            )
        if candidate.action_id in seen:
            continue
        seen.add(candidate.action_id)
        selected.append(candidate)
        if len(selected) >= top_k:
            break

    required_ids = _required_action_ids(selected, available, by_id, top_k=top_k)
    protected = set(required_ids)
    for action_id in required_ids:
        if any(candidate.action_id == action_id for candidate in selected):
            continue
        replacement = ActionCandidate(action_id=action_id)
        if len(selected) < top_k:
            selected.append(replacement)
            continue
        for index in range(len(selected) - 1, -1, -1):
            if selected[index].action_id not in protected:
                selected[index] = replacement
                break

    # Structural insertion may replace a low-ranked item, but never reorder surviving
    # policy-ranked candidates.  Required branches are appended/replaced deterministically.
    return selected[:top_k]


def _required_action_ids(
    selected: Sequence[ActionCandidate],
    available: Sequence[JsonObject],
    by_id: Mapping[str, JsonObject],
    *,
    top_k: int,
) -> list[str]:
    if top_k < 2:
        return []

    action_types = {str(action["action_type"]) for action in available}
    required: list[str] = []

    regular_combat = _SYSTEM_ACTION_TYPE in action_types and bool(
        action_types & {CARD_ACTION_TYPE, _POTION_ACTION_TYPE}
    )
    if regular_combat:
        system_id = _best_or_first_id(
            selected, available, by_id, {_SYSTEM_ACTION_TYPE}
        )
        card_id = _best_or_first_id(selected, available, by_id, {CARD_ACTION_TYPE})
        if system_id is not None:
            required.append(system_id)
        if card_id is not None:
            required.append(card_id)
        if top_k >= 3:
            potion_id = _best_or_first_id(
                selected, available, by_id, {_POTION_ACTION_TYPE}
            )
            if potion_id is not None:
                required.append(potion_id)

    if CHOICE_CARD_ACTION_TYPE in action_types:
        completion_id = _best_or_first_id(
            selected,
            available,
            by_id,
            {CHOICE_CONFIRM_ACTION_TYPE, CHOICE_SKIP_ACTION_TYPE},
        )
        if completion_id is not None:
            required.append(completion_id)

    # Preserve priority when top_k cannot fit every invariant (for example top_k=2 in
    # ordinary combat with potion available: End Turn + card win over potion).
    deduped: list[str] = []
    for action_id in required:
        if action_id not in deduped:
            deduped.append(action_id)
    return deduped[:top_k]


def _best_or_first_id(
    selected: Sequence[ActionCandidate],
    available: Sequence[JsonObject],
    by_id: Mapping[str, JsonObject],
    action_types: set[str],
) -> str | None:
    for candidate in selected:
        action = by_id.get(candidate.action_id)
        if action is not None and action.get("action_type") in action_types:
            return candidate.action_id
    for action in available:
        if action.get("action_type") in action_types:
            return str(action["action_id"])
    return None
