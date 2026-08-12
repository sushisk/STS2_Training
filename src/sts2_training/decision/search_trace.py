"""Combat search trace data structures for training and replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias

JsonObject = Mapping[str, Any]


@dataclass(frozen=True)
class SearchTraceStart:
    search_id: str
    instance_id: str
    root_decision_point_id: str
    beam_width: int
    top_k_actions: int
    max_depth: int
    max_continuation_steps: int
    time_budget_ms: float | None
    exhaustive_root_actions: bool
    pruner_name: str
    pruner_version: str
    event_type: str = field(default="search_start", init=False)


@dataclass(frozen=True)
class PolicyCandidateTrace:
    action_id: str
    action: JsonObject
    branch_id: str
    rng_id: int
    policy_rank: int | None
    policy_score: float | None
    post_coverage_rank: int
    candidate_source: str


@dataclass(frozen=True)
class PolicyProposalTrace:
    search_id: str
    proposal_step_id: str
    parent_node_id: str
    parent_branch_id: str
    decision_point_id: str
    legal_actions: tuple[JsonObject, ...]
    candidates: tuple[PolicyCandidateTrace, ...]
    requested_top_k: int
    exhaustive_root: bool
    event_type: str = field(default="policy_proposal", init=False)


@dataclass(frozen=True)
class ResolvedNodeTrace:
    """One emulator result materialized as a Beam node.

    ``value_source`` is ``terminal`` or ``value_bootstrap`` only when ``value_is_fresh``
    is true. Pending continuation nodes carry an inherited parent value and therefore use
    ``value_source='inherited'``; those values must not become learning targets.
    """

    search_id: str
    node_id: str
    parent_node_id: str
    branch_id: str
    parent_branch_id: str
    root_action_id: str | None
    rng_id: int
    decision_point_id: str
    depth: int
    combat_depth: int
    continuation_steps: int
    value: float
    value_is_fresh: bool
    value_source: str
    state_kind: str
    resolution: str
    terminal: bool
    action_id: str | None
    action_type: str | None
    action: JsonObject | None
    policy_rank: int | None
    policy_score: float | None
    post_coverage_rank: int | None
    candidate_source: str | None
    event_type: str = field(default="resolved_node", init=False)


@dataclass(frozen=True)
class StablePruneNodeTrace:
    node_id: str
    parent_node_id: str
    branch_id: str
    parent_branch_id: str
    frontier_index_before_prune: int
    kept: bool
    value: float
    root_action_id: str | None
    rng_id: int
    decision_point_id: str
    depth: int
    combat_depth: int
    continuation_steps: int
    terminal: bool
    action_id: str | None
    action_type: str | None
    action: JsonObject | None
    policy_rank: int | None
    policy_score: float | None
    post_coverage_rank: int | None
    candidate_source: str | None


@dataclass(frozen=True)
class StablePruneTrace:
    search_id: str
    prune_step_id: str
    phase: str
    k: int
    frontier_size: int
    pruner_name: str
    pruner_version: str
    max_depth: int
    depths_completed: int
    remaining_time_ms: float | None
    nodes: tuple[StablePruneNodeTrace, ...]
    event_type: str = field(default="stable_prune", init=False)


@dataclass(frozen=True)
class SearchTraceEnd:
    search_id: str
    reason: str
    best_root_action_id: str | None
    best_value: float | None
    depths_completed: int
    nodes_expanded: int
    branches_created: int
    event_type: str = field(default="search_end", init=False)


SearchTraceEvent: TypeAlias = (
    SearchTraceStart
    | PolicyProposalTrace
    | ResolvedNodeTrace
    | StablePruneTrace
    | SearchTraceEnd
)


class SearchTraceCollector(Protocol):
    def record(self, event: SearchTraceEvent) -> None:
        ...


class InMemorySearchTraceCollector:
    def __init__(self) -> None:
        self.events: list[SearchTraceEvent] = []

    def record(self, event: SearchTraceEvent) -> None:
        self.events.append(event)
