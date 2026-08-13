"""Structural candidate coverage applied after policy ranking.

A PolicyModel owns action preference/ranking. This module owns the small set of
search-topology invariants that should survive swapping the heuristic policy for a
learned one: retain a turn-boundary branch, retain at least one card branch in ordinary
combat, keep potion diversity when there is room, and retain a legal pending-choice
completion branch.

Phase-1 search instrumentation also needs to distinguish inner-policy ranking from
structural coverage. ``CandidateProposal`` carries that provenance without widening the
``ActionCandidate`` contract or changing candidate ordering. Canonically, the
pre-simulation scalar is an ``action_score``; the historical ``policy_score`` and
``policy_rank`` dataclass fields are retained as compatibility storage names.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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


@dataclass(frozen=True)
class CandidateProposal:
    """One post-coverage candidate plus action-ranking provenance for search traces."""

    candidate: ActionCandidate
    policy_rank: int | None
    policy_score: float | None
    post_coverage_rank: int
    candidate_source: str

    @property
    def action_id(self) -> str:
        return self.candidate.action_id

    @property
    def action_rank(self) -> int | None:
        """Canonical name for the candidate's pre-coverage rank."""

        return self.policy_rank

    @property
    def action_score(self) -> float | None:
        """Canonical name for the candidate's pre-simulation score."""

        return self.policy_score


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
        return [
            proposal.candidate
            for proposal in self.propose_with_provenance(
                legal_actions,
                masked_emulator_dto,
                top_k=top_k,
            )
        ]

    def propose_with_provenance(
        self,
        legal_actions: Sequence[JsonObject],
        masked_emulator_dto: Mapping[str, object],
        *,
        top_k: int,
    ) -> list[CandidateProposal]:
        ranked = self._inner.propose(
            legal_actions,
            masked_emulator_dto,
            top_k=top_k,
        )
        return apply_structural_coverage_with_provenance(
            ranked,
            legal_actions,
            top_k=top_k,
        )

    def propose_batch(
        self,
        requests: Sequence[
            tuple[Sequence[JsonObject], Mapping[str, object]]
        ],
        *,
        top_k: int,
    ) -> list[list[ActionCandidate]]:
        return [
            [proposal.candidate for proposal in proposals]
            for proposals in self.propose_batch_with_provenance(requests, top_k=top_k)
        ]

    def propose_batch_with_provenance(
        self,
        requests: Sequence[
            tuple[Sequence[JsonObject], Mapping[str, object]]
        ],
        *,
        top_k: int,
    ) -> list[list[CandidateProposal]]:
        ranked = self._inner.propose_batch(requests, top_k=top_k)
        if len(ranked) != len(requests):
            raise RuntimeError(
                "PolicyModel.propose_batch must return exactly one entry per request"
            )
        return [
            apply_structural_coverage_with_provenance(
                candidates,
                legal_actions,
                top_k=top_k,
            )
            for candidates, (legal_actions, _dto) in zip(ranked, requests)
        ]


def propose_batch_with_provenance(
    policy: PolicyModel,
    requests: Sequence[tuple[Sequence[JsonObject], Mapping[str, object]]],
    *,
    top_k: int,
) -> list[list[CandidateProposal]]:
    """Run any PolicyModel while retaining provenance when the adapter exposes it."""

    if isinstance(policy, CoverageConstrainedPolicy):
        return policy.propose_batch_with_provenance(requests, top_k=top_k)

    ranked = policy.propose_batch(requests, top_k=top_k)
    if len(ranked) != len(requests):
        raise RuntimeError("PolicyModel.propose_batch must return exactly one entry per request")
    return [
        _policy_only_proposals(candidates, top_k=top_k)
        for candidates in ranked
    ]


def apply_structural_coverage(
    ranked: Sequence[ActionCandidate],
    legal_actions: Sequence[JsonObject],
    *,
    top_k: int,
) -> list[ActionCandidate]:
    """Retain structural branches without changing the policy's relative ranking."""

    return [
        proposal.candidate
        for proposal in apply_structural_coverage_with_provenance(
            ranked,
            legal_actions,
            top_k=top_k,
        )
    ]


def apply_structural_coverage_with_provenance(
    ranked: Sequence[ActionCandidate],
    legal_actions: Sequence[JsonObject],
    *,
    top_k: int,
) -> list[CandidateProposal]:
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

    selected: list[CandidateProposal] = []
    seen: set[str] = set()
    for action_rank, candidate in enumerate(ranked):
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
        selected.append(
            CandidateProposal(
                candidate=candidate,
                policy_rank=action_rank,
                policy_score=_action_score(candidate),
                post_coverage_rank=len(selected),
                candidate_source="policy",
            )
        )
        if len(selected) >= top_k:
            break

    selected_candidates = [proposal.candidate for proposal in selected]
    required_ids = _required_action_ids(
        selected_candidates,
        available,
        by_id,
        top_k=top_k,
    )
    protected = set(required_ids)
    for action_id in required_ids:
        if any(proposal.action_id == action_id for proposal in selected):
            continue
        replacement = CandidateProposal(
            candidate=ActionCandidate(action_id=action_id),
            policy_rank=None,
            policy_score=None,
            post_coverage_rank=-1,
            candidate_source="structural_coverage",
        )
        if len(selected) < top_k:
            selected.append(replacement)
            continue
        for index in range(len(selected) - 1, -1, -1):
            if selected[index].action_id not in protected:
                selected[index] = replacement
                break

    return [
        CandidateProposal(
            candidate=proposal.candidate,
            policy_rank=proposal.action_rank,
            policy_score=proposal.action_score,
            post_coverage_rank=index,
            candidate_source=proposal.candidate_source,
        )
        for index, proposal in enumerate(selected[:top_k])
    ]


def _policy_only_proposals(
    ranked: Sequence[ActionCandidate],
    *,
    top_k: int,
) -> list[CandidateProposal]:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    proposals: list[CandidateProposal] = []
    for action_rank, candidate in enumerate(ranked):
        if not isinstance(candidate, ActionCandidate):
            raise RuntimeError("PolicyModel.propose_batch must return ActionCandidate objects")
        proposals.append(
            CandidateProposal(
                candidate=candidate,
                policy_rank=action_rank,
                policy_score=_action_score(candidate),
                post_coverage_rank=action_rank,
                candidate_source="policy",
            )
        )
    return proposals


def _action_score(candidate: ActionCandidate) -> float | None:
    """Read a finite pre-simulation action score from known candidate conventions."""

    for attribute in ("action_score", "policy_score", "score", "prior", "logit"):
        raw = getattr(candidate, attribute, None)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        action_score = float(raw)
        if math.isfinite(action_score):
            return action_score
    return None


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
            selected,
            available,
            by_id,
            {_SYSTEM_ACTION_TYPE},
        )
        card_id = _best_or_first_id(selected, available, by_id, {CARD_ACTION_TYPE})
        if system_id is not None:
            required.append(system_id)
        if card_id is not None:
            required.append(card_id)
        if top_k >= 3:
            potion_id = _best_or_first_id(
                selected,
                available,
                by_id,
                {_POTION_ACTION_TYPE},
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
