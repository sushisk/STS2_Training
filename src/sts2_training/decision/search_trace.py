"""Combat search trace data structures for training and replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias

from sts2_training.decision.stable_pruner import StablePruneContext, StablePruneNodeView

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
    pruner_name: str
    pruner_version: str
    exhaustive_root_actions: bool = False
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
    requested_top_k: int = 0
    exhaustive_root: bool = False
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

    def to_prune_view(self) -> StablePruneNodeView:
        """Reconstruct exactly the public pruning view for this trace node."""

        return StablePruneNodeView(
            value=self.value,
            root_action_id=self.root_action_id,
            depth=self.depth,
            combat_depth=self.combat_depth,
            continuation_steps=self.continuation_steps,
            terminal=self.terminal,
            action_type=self.action_type,
            policy_rank=self.policy_rank,
            policy_score=self.policy_score,
            post_coverage_rank=self.post_coverage_rank,
            candidate_source=self.candidate_source,
        )


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

    def node_views(self) -> tuple[StablePruneNodeView, ...]:
        """Return runtime-equivalent public views in authoritative frontier order."""

        if len(self.nodes) != self.frontier_size:
            raise ValueError("StablePruneTrace frontier_size must equal len(nodes)")
        views: list[StablePruneNodeView] = []
        for index, node in enumerate(self.nodes):
            if node.frontier_index_before_prune != index:
                raise ValueError(
                    "StablePruneTrace nodes must be stored in authoritative frontier order"
                )
            views.append(node.to_prune_view())
        return tuple(views)

    def to_prune_context(
        self,
        *,
        beam_width: int | None = None,
    ) -> StablePruneContext:
        """Reconstruct pruning context, optionally overriding the target beam width.

        The default uses trace ``k``. Offline counterfactual replay can pass a narrower
        target beam width while keeping every other runtime context field unchanged.
        """

        return StablePruneContext(
            search_id=self.search_id,
            prune_step_id=self.prune_step_id,
            phase=self.phase,
            beam_width=self.k if beam_width is None else beam_width,
            max_depth=self.max_depth,
            depths_completed=self.depths_completed,
            remaining_time_ms=self.remaining_time_ms,
        )


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
