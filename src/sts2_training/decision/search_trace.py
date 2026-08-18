"""Combat search trace data structures for training and replay.

The persisted v4 field spellings are retained for record compatibility, while canonical
Python access follows the Oracle terminology: ``action_score``/``action_rank`` for
pre-simulation policy provenance, ``state_score`` for a resolved state, and
``best_node_score`` for the score attributed to the selected search node.
"""

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

    @property
    def action_rank(self) -> int | None:
        return self.policy_rank

    @property
    def action_score(self) -> float | None:
        return self.policy_score


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

    Canonically, ``state_score_source`` is ``terminal`` or ``value_bootstrap`` only when
    ``state_score_is_fresh`` is true. Pending continuation nodes carry an inherited parent
    state score and therefore use ``state_score_source='inherited'``; those scores must not
    become learning targets. The stored v4 names remain ``value``/``value_is_fresh``/
    ``value_source`` for trace compatibility.
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

    @property
    def state_score(self) -> float:
        return self.value

    @property
    def state_score_is_fresh(self) -> bool:
        return self.value_is_fresh

    @property
    def state_score_source(self) -> str:
        return self.value_source

    @property
    def action_rank(self) -> int | None:
        return self.policy_rank

    @property
    def action_score(self) -> float | None:
        return self.policy_score


@dataclass(frozen=True)
class BranchFaultTrace:
    """A proposed branch whose `emulate_actions` result never became a Beam node.

    Recorded at the exact point `BeamSearchEngine._score_frontier` would otherwise drop
    the branch silently - whenever the frontier still has other branches that did resolve,
    nothing else in the search loop notices this branch's fate. In memory, ``detail`` keeps
    the RL error text verbatim for target derivation/audit. Oracle JSONL serialization may
    persist full detail only for the first repeated signature while retaining every trace's
    lineage/censoring fields and adding an aggregate to the persisted search-end payload.
    """

    search_id: str
    node_id: str
    parent_node_id: str
    branch_id: str
    parent_branch_id: str
    root_action_id: str | None
    rng_id: int
    depth: int
    combat_depth: int
    continuation_steps: int
    action_id: str
    action_type: str | None
    action: JsonObject | None
    policy_rank: int | None
    policy_score: float | None
    post_coverage_rank: int | None
    candidate_source: str | None
    status: str
    fault_kind: str | None
    detail: str | None
    event_type: str = field(default="branch_fault", init=False)

    @property
    def action_rank(self) -> int | None:
        return self.policy_rank

    @property
    def action_score(self) -> float | None:
        return self.policy_score


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

    @property
    def state_score(self) -> float:
        return self.value

    @property
    def action_rank(self) -> int | None:
        return self.policy_rank

    @property
    def action_score(self) -> float | None:
        return self.policy_score

    def to_prune_view(self) -> StablePruneNodeView:
        """Reconstruct exactly the public pruning view for this trace node."""

        return StablePruneNodeView(
            value=self.state_score,
            root_action_id=self.root_action_id,
            depth=self.depth,
            combat_depth=self.combat_depth,
            continuation_steps=self.continuation_steps,
            terminal=self.terminal,
            action_type=self.action_type,
            policy_rank=self.action_rank,
            policy_score=self.action_score,
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
    # Ordered indices returned by StableFrontierPruner.select(). Runtime/v4 records always
    # set this tuple explicitly. None is retained only for programmatic legacy constructors
    # that need frontier-view reconstruction but cannot claim exact survivor-order replay.
    selected_indices: tuple[int, ...] | None = None
    event_type: str = field(default="stable_prune", init=False)

    def node_views(self) -> tuple[StablePruneNodeView, ...]:
        """Return runtime-equivalent public views in authoritative frontier order."""

        if len(self.nodes) != self.frontier_size:
            raise ValueError("StablePruneTrace frontier_size must equal len(nodes)")

        selected_set: set[int] | None = None
        if self.selected_indices is not None:
            if len(self.selected_indices) > self.k:
                raise ValueError(
                    "StablePruneTrace selected_indices cannot contain more than k entries"
                )
            selected_set = set()
            for index in self.selected_indices:
                if isinstance(index, bool) or not isinstance(index, int):
                    raise ValueError("StablePruneTrace selected_indices must contain integers")
                if index < 0 or index >= self.frontier_size:
                    raise ValueError(
                        "StablePruneTrace selected_indices contain an out-of-range index"
                    )
                if index in selected_set:
                    raise ValueError("StablePruneTrace selected_indices must be unique")
                selected_set.add(index)

        views: list[StablePruneNodeView] = []
        for index, node in enumerate(self.nodes):
            if node.frontier_index_before_prune != index:
                raise ValueError(
                    "StablePruneTrace nodes must be stored in authoritative frontier order"
                )
            if selected_set is not None and node.kept != (index in selected_set):
                raise ValueError(
                    "StablePruneTrace kept membership must match ordered selected_indices"
                )
            views.append(node.to_prune_view())
        return tuple(views)

    def selected_node_views(self) -> tuple[StablePruneNodeView, ...]:
        """Reconstruct survivors in the exact order returned by the runtime pruner."""

        if self.selected_indices is None:
            raise ValueError(
                "StablePruneTrace survivor order is unavailable; selected_indices is required"
            )
        views = self.node_views()
        return tuple(views[index] for index in self.selected_indices)

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
    branches_faulted: int = 0
    # Optional additive diagnostic for callers that build an aggregate in memory. Oracle
    # JSONL serialization also fills the persisted payload without mutating this event.
    fault_summaries: tuple[JsonObject, ...] = ()
    event_type: str = field(default="search_end", init=False)

    @property
    def best_node_score(self) -> float | None:
        """Canonical name for the score attributed to the selected search-tree node."""

        return self.best_value


SearchTraceEvent: TypeAlias = (
    SearchTraceStart
    | PolicyProposalTrace
    | ResolvedNodeTrace
    | BranchFaultTrace
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
