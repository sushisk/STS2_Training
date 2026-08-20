"""Budgeted Combat oracle collection and target derivation.

The oracle is deliberately an expensive *estimate*, not ground truth. It runs the same
BeamSearch traversal with a larger budget, records all resolved nodes that the base search
materializes, and derives targets with explicit censoring metadata.

Score naming follows the Oracle contract: ValueModel emits a ``state_score`` for one
resolved state; once that scalar is attributed to a concrete search-tree branch/action it
is a ``node_score``. Historical persisted fields such as ``value``, ``target_value`` and
``estimated_q`` remain compatibility spellings, with canonical Python properties exposed
on the target dataclasses.

Ordinary root actions keep independent RNG hypotheses. Cross-action common-RNG sampling
is intentionally unsupported until the Training/RL API contract explicitly guarantees
its semantics.
"""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from statistics import fmean
from typing import Any

from sts2_training.api.contract import ROOT_BRANCH_ID
from sts2_training.decision.beam_search import (
    BeamNode,
    BeamSearchConfig,
    BeamSearchEngine,
    BeamSearchResult,
    BranchIdAllocator,
)
from sts2_training.decision.candidate_coverage import CoverageConstrainedPolicy
from sts2_training.decision.combat_decision import is_continuation_decision
from sts2_training.decision.policy import PolicyModel
from sts2_training.decision.search_trace import (
    BranchFaultTrace,
    InMemorySearchTraceCollector,
    PolicyProposalTrace,
    ResolvedNodeTrace,
    SearchTraceEnd,
    SearchTraceEvent,
    SearchTraceStart,
    StablePruneTrace,
)
from sts2_training.decision.stable_pruner import StableFrontierPruner, ValueTopKPruner
from sts2_training.decision.value import ValueModel

JsonObject = Mapping[str, Any]


@dataclass(frozen=True)
class OracleCollectionConfig:
    """Explicit search budget and target budget for one oracle collection pass."""

    beam_config: BeamSearchConfig = field(
        default_factory=lambda: BeamSearchConfig(
            beam_width=32,
            top_k_actions=8,
            max_depth=4,
        )
    )
    target_beam_width: int = 8
    exhaustive_root_actions: bool = True
    rng_sampling: str = "independent"

    def __post_init__(self) -> None:
        if (
            isinstance(self.target_beam_width, bool)
            or not isinstance(self.target_beam_width, int)
            or self.target_beam_width <= 0
        ):
            raise ValueError("target_beam_width must be a positive integer")
        if self.beam_config.beam_width < self.target_beam_width:
            raise ValueError("oracle beam_width must be >= target_beam_width")
        if not isinstance(self.exhaustive_root_actions, bool):
            raise TypeError("exhaustive_root_actions must be a bool")
        if self.rng_sampling != "independent":
            raise ValueError(
                "only rng_sampling='independent' is supported until cross-action "
                "common-RNG semantics are explicitly verified by the Training/RL API"
            )


@dataclass(frozen=True)
class OracleRngOutcome:
    rng_id: int
    value: float | None
    target_source: str
    terminal_reached: bool
    deepest_combat_depth: int
    censored: bool
    censor_reason: str | None
    best_node_id: str | None

    @property
    def node_score(self) -> float | None:
        """Canonical name for this RNG hypothesis's attributed node score."""

        return self.value


@dataclass(frozen=True)
class RootActionOracleTarget:
    action_id: str
    action: JsonObject
    evaluated: bool
    estimated_q: float | None
    rng_outcomes: tuple[OracleRngOutcome, ...]
    target_source: str
    terminal_reached: bool
    censored: bool
    censor_reason: str | None

    @property
    def node_score(self) -> float | None:
        """Canonical name for the root action's aggregated Oracle node score."""

        return self.estimated_q


@dataclass(frozen=True)
class StableNodeOracleTarget:
    prune_step_id: str
    node_id: str
    root_action_id: str | None
    frontier_index_before_prune: int
    oracle_kept: bool
    target_beam_width: int
    baseline_would_keep: bool
    target_value: float | None
    target_source: str
    terminal_reached: bool
    censored: bool
    censor_reason: str | None
    best_descendant_node_id: str | None

    @property
    def target_node_score(self) -> float | None:
        """Canonical name for the descendant-derived node-score target."""

        return self.target_value


@dataclass(frozen=True)
class OracleTargetMetadata:
    search_id: str
    oracle_beam_width: int
    target_beam_width: int
    top_k_actions: int
    max_depth: int
    max_continuation_steps: int
    time_budget_ms: float | None
    exhaustive_root_actions: bool
    rng_sampling: str
    search_reason: str
    pruner_name: str
    pruner_version: str


@dataclass(frozen=True)
class OracleTargets:
    metadata: OracleTargetMetadata
    root_actions: tuple[RootActionOracleTarget, ...]
    stable_nodes: tuple[StableNodeOracleTarget, ...]


@dataclass(frozen=True)
class OracleProvenance:
    """Identity and configuration of the actual teacher inference stack."""

    teacher_policy_class: str
    teacher_inner_policy_class: str
    teacher_coverage_policy_class: str | None
    teacher_value_class: str
    teacher_policy_metadata: JsonObject = field(default_factory=dict)
    teacher_inner_policy_metadata: JsonObject = field(default_factory=dict)
    teacher_value_metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class OracleCollectionResult:
    search_result: BeamSearchResult
    trace: tuple[SearchTraceEvent, ...]
    targets: OracleTargets
    provenance: OracleProvenance


class _OracleTraceCollector(InMemorySearchTraceCollector):
    def __init__(
        self,
        *,
        descendant_top_k: int,
        exhaustive_root_actions: bool,
        effective_time_budget_ms: float,
    ) -> None:
        super().__init__()
        self._descendant_top_k = descendant_top_k
        self._exhaustive_root_actions = exhaustive_root_actions
        self._effective_time_budget_ms = effective_time_budget_ms

    def record(self, event: SearchTraceEvent) -> None:
        if isinstance(event, SearchTraceStart):
            event = replace(
                event,
                time_budget_ms=self._effective_time_budget_ms,
                exhaustive_root_actions=self._exhaustive_root_actions,
            )
        elif isinstance(event, PolicyProposalTrace):
            is_root = event.parent_branch_id == ROOT_BRANCH_ID
            requested_top_k = self._descendant_top_k
            if is_root and self._exhaustive_root_actions:
                requested_top_k = len(_available_trace_actions(event.legal_actions))
            event = replace(
                event,
                requested_top_k=requested_top_k,
                exhaustive_root=is_root and self._exhaustive_root_actions,
            )
        super().record(event)


class _OracleBeamSearchEngine(BeamSearchEngine):
    """Training-only BeamSearch extension that exposes resolved-node outcomes."""

    def __init__(self, *args: Any, exhaustive_root_actions: bool, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._exhaustive_root_actions = exhaustive_root_actions

    def _propose_frontier(self, beam, *, search_id, proposal_step_index):  # type: ignore[override]
        is_root = len(beam) == 1 and beam[0].branch_id == ROOT_BRANCH_ID
        if not (is_root and self._exhaustive_root_actions):
            return super()._propose_frontier(
                beam,
                search_id=search_id,
                proposal_step_index=proposal_step_index,
            )

        available_count = len(_available_dto_actions(beam[0].masked_emulator_dto))
        original_top_k = self.config.top_k_actions
        self.config.top_k_actions = max(original_top_k, available_count)
        try:
            result = super()._propose_frontier(
                beam,
                search_id=search_id,
                proposal_step_index=proposal_step_index,
            )
        finally:
            self.config.top_k_actions = original_top_k

        _items, item_meta, _policy_ms, _exhausted = result
        proposed_ids = {proposal.action_id for _node, proposal, _branch, _rng in item_meta}
        available_ids = {
            str(action["action_id"])
            for action in _available_dto_actions(beam[0].masked_emulator_dto)
        }
        if proposed_ids != available_ids:
            missing = sorted(available_ids - proposed_ids)
            raise RuntimeError(
                "exhaustive root collection requires PolicyModel to return every "
                f"available action when requested; missing={missing!r}"
            )
        return result

    def _score_frontier(  # type: ignore[override]
        self, item_meta, branch_results, depth=None, *, search_id: str, root_turn=None
    ):
        result = super()._score_frontier(
            item_meta, branch_results, depth, search_id=search_id, root_turn=root_turn
        )
        next_beam, newly_finished = result[0], result[1]
        if self.trace_collector is not None:
            finished_object_ids = {id(node) for node in newly_finished}
            for node in (*next_beam, *newly_finished):
                continuation = is_continuation_decision(node.masked_emulator_dto)
                state_score_is_fresh = not continuation
                trace_state_score = node.value
                if node.terminal:
                    state_kind = "terminal"
                    resolution = "terminal"
                    exact_terminal = self._value_fn.exact_terminal_utility(
                        node.masked_emulator_dto
                    )
                    if exact_terminal is None:
                        state_score_source = "value_bootstrap"
                    else:
                        if (
                            isinstance(exact_terminal, bool)
                            or not isinstance(exact_terminal, (int, float))
                            or not math.isfinite(float(exact_terminal))
                        ):
                            raise RuntimeError(
                                "ValueModel.exact_terminal_utility must return a finite "
                                "number or None"
                            )
                        trace_state_score = float(exact_terminal)
                        state_score_source = "terminal"
                elif continuation:
                    state_kind = "continuation"
                    resolution = "continuation"
                    state_score_source = "inherited"
                else:
                    state_kind = "stable"
                    state_score_source = "value_bootstrap"
                    if id(node) in finished_object_ids:
                        resolution = (
                            "max_depth"
                            if node.combat_depth >= self.config.max_depth
                            else "cannot_expand"
                        )
                    else:
                        resolution = "expandable_stable"
                self.trace_collector.record(
                    ResolvedNodeTrace(
                        search_id=search_id,
                        node_id=f"{search_id}:{node.branch_id}",
                        parent_node_id=f"{search_id}:{node.parent_branch_id}",
                        branch_id=node.branch_id,
                        parent_branch_id=node.parent_branch_id,
                        root_action_id=node.root_action_id,
                        rng_id=node.rng_id,
                        decision_point_id=node.decision_point_id,
                        depth=node.depth,
                        combat_depth=node.combat_depth,
                        continuation_steps=node.continuation_steps,
                        value=trace_state_score,
                        value_is_fresh=state_score_is_fresh,
                        value_source=state_score_source,
                        state_kind=state_kind,
                        resolution=resolution,
                        terminal=node.terminal,
                        action_id=node.action_id,
                        action_type=node.action_type,
                        action=None if node.action is None else dict(node.action),
                        policy_rank=node.policy_rank,
                        policy_score=node.policy_score,
                        post_coverage_rank=node.post_coverage_rank,
                        candidate_source=node.candidate_source,
                    )
                )
        return result


class BudgetedOracleCollector:
    """Run an expensive BeamSearch estimate and derive training targets from its trace."""

    def __init__(
        self,
        client: Any,
        *,
        policy: PolicyModel,
        value_fn: ValueModel,
        config: OracleCollectionConfig | None = None,
        stable_pruner: StableFrontierPruner | None = None,
        branch_allocator: BranchIdAllocator | None = None,
    ) -> None:
        self._client = client
        self._policy = policy
        self._value_fn = value_fn
        self.config = config or OracleCollectionConfig()
        self._stable_pruner = stable_pruner or ValueTopKPruner()
        self._branch_allocator = branch_allocator or BranchIdAllocator()

    @classmethod
    def from_beam_engine(
        cls,
        engine: BeamSearchEngine,
        *,
        config: OracleCollectionConfig | None = None,
        stable_pruner: StableFrontierPruner | None = None,
    ) -> "BudgetedOracleCollector":
        policy = _independent_policy_copy(engine._policy)  # noqa: SLF001
        value_fn = _independent_value_copy(engine._value_fn)  # noqa: SLF001
        return cls(
            engine._client,  # noqa: SLF001 - same-package training instrumentation
            policy=policy,
            value_fn=value_fn,
            config=config,
            stable_pruner=stable_pruner,
            branch_allocator=engine._allocator,  # noqa: SLF001 - shared per-instance namespace
        )

    async def collect(
        self,
        instance_id: str,
        root_decision: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> OracleCollectionResult:
        beam_config = replace(
            self.config.beam_config,
            simulation_options=(
                None
                if self.config.beam_config.simulation_options is None
                else dict(self.config.beam_config.simulation_options)
            ),
        )
        trace_collector = _OracleTraceCollector(
            descendant_top_k=beam_config.top_k_actions,
            exhaustive_root_actions=self.config.exhaustive_root_actions,
            effective_time_budget_ms=_effective_time_budget_ms(
                timeout_s,
                beam_config.time_budget_ms,
            ),
        )
        provenance = _oracle_provenance(self._policy, self._value_fn)
        engine = _OracleBeamSearchEngine(
            self._client,
            policy=self._policy,
            value_fn=self._value_fn,
            config=beam_config,
            stable_pruner=self._stable_pruner,
            trace_collector=trace_collector,
            exhaustive_root_actions=self.config.exhaustive_root_actions,
        )
        # One allocator namespace is shared across Oracle passes and, when constructed
        # from the runtime BeamSearchEngine, across teacher/runtime searches as well.
        # This prevents RL-side RNG hypothesis identity from being accidentally reused.
        engine._allocator = self._branch_allocator  # noqa: SLF001
        result = await engine.search(
            instance_id,
            root_decision,
            timeout_s=timeout_s,
        )
        targets = build_oracle_targets(
            trace_collector.events,
            target_beam_width=self.config.target_beam_width,
            exhaustive_root_actions=self.config.exhaustive_root_actions,
            rng_sampling=self.config.rng_sampling,
        )
        return OracleCollectionResult(
            search_result=result,
            trace=tuple(trace_collector.events),
            targets=targets,
            provenance=provenance,
        )


def build_oracle_targets(
    events: Sequence[SearchTraceEvent],
    *,
    target_beam_width: int,
    exhaustive_root_actions: bool,
    rng_sampling: str = "independent",
) -> OracleTargets:
    """Derive root-action and stable-pruning node-score targets from one oracle trace.

    Stable-node targets use *future leaf descendants only*. A node that the oracle itself
    pruned before follow-up therefore receives ``no_target`` rather than its current
    ValueModel state score, avoiding direct self-imitation of the pruning rule.
    Intermediate states that were subsequently expanded are not treated as final node
    outcomes either. A node with a recorded policy proposal is also not a final leaf when
    that expansion attempt failed to materialize a resolved child. Any recorded branch
    fault makes the affected RNG/stable subtree a ``no_target``: a surviving sibling does
    not prove that the missing/faulted candidate would have scored worse.
    """

    starts = [event for event in events if isinstance(event, SearchTraceStart)]
    ends = [event for event in events if isinstance(event, SearchTraceEnd)]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("oracle target derivation requires exactly one search_start/search_end")
    start = starts[0]
    end = ends[0]
    if (
        isinstance(target_beam_width, bool)
        or not isinstance(target_beam_width, int)
        or target_beam_width <= 0
    ):
        raise ValueError("target_beam_width must be a positive integer")
    if target_beam_width > start.beam_width:
        raise ValueError("target_beam_width cannot exceed oracle beam_width")

    resolved = [event for event in events if isinstance(event, ResolvedNodeTrace)]
    faults = [event for event in events if isinstance(event, BranchFaultTrace)]
    prune_events = [event for event in events if isinstance(event, StablePruneTrace)]
    proposal_events = [event for event in events if isinstance(event, PolicyProposalTrace)]
    root_proposals = [
        event
        for event in proposal_events
        if event.parent_branch_id == ROOT_BRANCH_ID
    ]
    if len(root_proposals) != 1:
        raise ValueError("oracle target derivation requires exactly one root policy proposal")
    root_proposal = root_proposals[0]

    resolved_by_id = {node.node_id: node for node in resolved}
    children: dict[str, list[str]] = defaultdict(list)
    for node in resolved:
        children[node.parent_node_id].append(node.node_id)
    oracle_pruned_node_ids = {
        node.node_id
        for prune in prune_events
        for node in prune.nodes
        if not node.kept
    }
    expanded_node_ids = {proposal.parent_node_id for proposal in proposal_events}

    fault_reason_by_root_rng: dict[tuple[str, int], str] = {}
    fault_depth_by_root_rng: dict[tuple[str, int], int] = {}
    fault_reason_by_ancestor: dict[str, str] = {}
    for fault in faults:
        reason = _branch_fault_censor_reason(fault)
        if fault.root_action_id is not None:
            key = (fault.root_action_id, fault.rng_id)
            fault_reason_by_root_rng.setdefault(key, reason)
            fault_depth_by_root_rng[key] = max(
                fault_depth_by_root_rng.get(key, 0),
                fault.combat_depth,
            )
        ancestor_id = fault.parent_node_id
        seen_ancestor_ids: set[str] = set()
        while ancestor_id in resolved_by_id and ancestor_id not in seen_ancestor_ids:
            seen_ancestor_ids.add(ancestor_id)
            fault_reason_by_ancestor.setdefault(ancestor_id, reason)
            ancestor_id = resolved_by_id[ancestor_id].parent_node_id

    root_targets = _root_action_targets(
        root_proposal,
        resolved,
        children=children,
        oracle_pruned_node_ids=oracle_pruned_node_ids,
        expanded_node_ids=expanded_node_ids,
        fault_reason_by_root_rng=fault_reason_by_root_rng,
        fault_depth_by_root_rng=fault_depth_by_root_rng,
        exhaustive_root_actions=exhaustive_root_actions,
        search_reason=end.reason,
    )
    stable_targets: list[StableNodeOracleTarget] = []
    for prune in prune_events:
        baseline_ids = _state_score_top_k_ids(prune, k=target_beam_width)
        for node in prune.nodes:
            descendants = _descendant_nodes(
                node.node_id,
                children=children,
                resolved_by_id=resolved_by_id,
            )
            raw_leaf_descendants = _leaf_fresh_nodes(descendants, children=children)
            leaf_descendants = [
                candidate
                for candidate in raw_leaf_descendants
                if candidate.node_id not in oracle_pruned_node_ids
                and candidate.node_id not in expanded_node_ids
            ]
            best = max(
                leaf_descendants,
                key=lambda candidate: candidate.state_score,
                default=None,
            )
            branch_fault_reason = fault_reason_by_ancestor.get(node.node_id)
            if branch_fault_reason is not None:
                target_node_score = None
                target_source = "no_target"
                terminal_reached = any(candidate.terminal for candidate in leaf_descendants)
                censored = True
                censor_reason = branch_fault_reason
                best_node_id = None
            elif best is None:
                target_node_score = None
                target_source = "no_target"
                terminal_reached = False
                censored = True
                censor_reason = (
                    "oracle_pruned_before_followup"
                    if not node.kept
                    else (
                        "oracle_descendant_pruned_before_followup"
                        if any(
                            candidate.node_id in oracle_pruned_node_ids
                            for candidate in raw_leaf_descendants
                        )
                        else (
                            f"oracle_descendant_expansion_without_outcome:{end.reason}"
                            if any(
                                candidate.node_id in expanded_node_ids
                                for candidate in raw_leaf_descendants
                            )
                            else f"search_ended_before_followup:{end.reason}"
                        )
                    )
                )
                best_node_id = None
            else:
                target_node_score = best.state_score
                target_source = best.state_score_source
                terminal_reached = any(candidate.terminal for candidate in leaf_descendants)
                censored = best.state_score_source != "terminal"
                censor_reason = None if not censored else f"value_bootstrap:{best.resolution}"
                best_node_id = best.node_id
            stable_targets.append(
                StableNodeOracleTarget(
                    prune_step_id=prune.prune_step_id,
                    node_id=node.node_id,
                    root_action_id=node.root_action_id,
                    frontier_index_before_prune=node.frontier_index_before_prune,
                    oracle_kept=node.kept,
                    target_beam_width=target_beam_width,
                    baseline_would_keep=node.node_id in baseline_ids,
                    target_value=target_node_score,
                    target_source=target_source,
                    terminal_reached=terminal_reached,
                    censored=censored,
                    censor_reason=censor_reason,
                    best_descendant_node_id=best_node_id,
                )
            )

    return OracleTargets(
        metadata=OracleTargetMetadata(
            search_id=start.search_id,
            oracle_beam_width=start.beam_width,
            target_beam_width=target_beam_width,
            top_k_actions=start.top_k_actions,
            max_depth=start.max_depth,
            max_continuation_steps=start.max_continuation_steps,
            time_budget_ms=start.time_budget_ms,
            exhaustive_root_actions=exhaustive_root_actions,
            rng_sampling=rng_sampling,
            search_reason=end.reason,
            pruner_name=start.pruner_name,
            pruner_version=start.pruner_version,
        ),
        root_actions=tuple(root_targets),
        stable_nodes=tuple(stable_targets),
    )


def _root_action_targets(
    root_proposal: PolicyProposalTrace,
    resolved: Sequence[ResolvedNodeTrace],
    *,
    children: Mapping[str, Sequence[str]],
    oracle_pruned_node_ids: set[str],
    expanded_node_ids: set[str],
    fault_reason_by_root_rng: Mapping[tuple[str, int], str],
    fault_depth_by_root_rng: Mapping[tuple[str, int], int],
    exhaustive_root_actions: bool,
    search_reason: str,
) -> list[RootActionOracleTarget]:
    available_actions = _available_trace_actions(root_proposal.legal_actions)
    candidates_by_id = {candidate.action_id: candidate for candidate in root_proposal.candidates}
    nodes_by_key: dict[tuple[str, int], list[ResolvedNodeTrace]] = defaultdict(list)
    for node in resolved:
        if node.root_action_id is not None:
            nodes_by_key[(node.root_action_id, node.rng_id)].append(node)

    targets: list[RootActionOracleTarget] = []
    for action in available_actions:
        action_id = str(action["action_id"])
        candidate = candidates_by_id.get(action_id)
        if candidate is None:
            reason = (
                "exhaustive_root_missing_candidate"
                if exhaustive_root_actions
                else "policy_candidate_limit"
            )
            targets.append(
                RootActionOracleTarget(
                    action_id=action_id,
                    action=dict(action),
                    evaluated=False,
                    estimated_q=None,
                    rng_outcomes=(),
                    target_source="no_target",
                    terminal_reached=False,
                    censored=True,
                    censor_reason=reason,
                )
            )
            continue

        resolved_rng_ids = {
            rng_id
            for root_action_id, rng_id in nodes_by_key
            if root_action_id == action_id
        }
        fault_rng_ids = {
            rng_id
            for root_action_id, rng_id in fault_reason_by_root_rng
            if root_action_id == action_id
        }
        if not resolved_rng_ids and not fault_rng_ids:
            if exhaustive_root_actions:
                raise RuntimeError(
                    "exhaustive root collection requires every proposed available action "
                    "to materialize at least one resolved outcome; "
                    f"action_id={action_id!r}, search_reason={search_reason!r}"
                )
            targets.append(
                RootActionOracleTarget(
                    action_id=action_id,
                    action=dict(action),
                    evaluated=False,
                    estimated_q=None,
                    rng_outcomes=(),
                    target_source="no_target",
                    terminal_reached=False,
                    censored=True,
                    censor_reason=f"candidate_not_materialized:{search_reason}",
                )
            )
            continue

        rng_ids = sorted(resolved_rng_ids | fault_rng_ids)
        outcomes: list[OracleRngOutcome] = []
        for rng_id in rng_ids:
            nodes = nodes_by_key.get((action_id, rng_id), [])
            raw_leaves = _leaf_fresh_nodes(nodes, children=children)
            leaves = [
                node
                for node in raw_leaves
                if node.node_id not in oracle_pruned_node_ids
                and node.node_id not in expanded_node_ids
            ]
            best = max(leaves, key=lambda node: node.state_score, default=None)
            deepest = max(
                max((node.combat_depth for node in nodes), default=0),
                fault_depth_by_root_rng.get((action_id, rng_id), 0),
            )
            terminal_reached = any(node.terminal for node in leaves)
            branch_fault_reason = fault_reason_by_root_rng.get((action_id, rng_id))
            if branch_fault_reason is not None:
                outcomes.append(
                    OracleRngOutcome(
                        rng_id=rng_id,
                        value=None,
                        target_source="no_target",
                        terminal_reached=terminal_reached,
                        deepest_combat_depth=deepest,
                        censored=True,
                        censor_reason=branch_fault_reason,
                        best_node_id=None,
                    )
                )
            elif best is None:
                censor_reason = (
                    "oracle_pruned_before_followup"
                    if any(
                        node.node_id in oracle_pruned_node_ids for node in raw_leaves
                    )
                    else (
                        f"expansion_attempted_without_outcome:{search_reason}"
                        if any(node.node_id in expanded_node_ids for node in raw_leaves)
                        else f"no_fresh_leaf_outcome:{search_reason}"
                    )
                )
                outcomes.append(
                    OracleRngOutcome(
                        rng_id=rng_id,
                        value=None,
                        target_source="no_target",
                        terminal_reached=False,
                        deepest_combat_depth=deepest,
                        censored=True,
                        censor_reason=censor_reason,
                        best_node_id=None,
                    )
                )
            else:
                censored = best.state_score_source != "terminal"
                outcomes.append(
                    OracleRngOutcome(
                        rng_id=rng_id,
                        value=best.state_score,
                        target_source=best.state_score_source,
                        terminal_reached=terminal_reached,
                        deepest_combat_depth=deepest,
                        censored=censored,
                        censor_reason=(
                            None if not censored else f"value_bootstrap:{best.resolution}"
                        ),
                        best_node_id=best.node_id,
                    )
                )

        node_scores = [
            outcome.node_score for outcome in outcomes if outcome.node_score is not None
        ]
        sources = {
            outcome.target_source for outcome in outcomes if outcome.node_score is not None
        }
        branch_fault_reason = next(
            (
                outcome.censor_reason
                for outcome in outcomes
                if outcome.censor_reason is not None
                and outcome.censor_reason.startswith("branch_fault:")
            ),
            None,
        )
        if branch_fault_reason is not None:
            target_source = "no_target"
            estimated_q = None
        elif not node_scores:
            target_source = "no_target"
            estimated_q = None
        elif sources == {"terminal"}:
            target_source = "terminal"
            estimated_q = fmean(node_scores)
        elif sources == {"value_bootstrap"}:
            target_source = "value_bootstrap"
            estimated_q = fmean(node_scores)
        else:
            target_source = "mixed"
            estimated_q = fmean(node_scores)
        censored = any(outcome.censored for outcome in outcomes) or not node_scores
        censor_reason = branch_fault_reason or next(
            (outcome.censor_reason for outcome in outcomes if outcome.censor_reason is not None),
            None,
        )
        targets.append(
            RootActionOracleTarget(
                action_id=action_id,
                action=dict(action),
                evaluated=True,
                estimated_q=estimated_q,
                rng_outcomes=tuple(outcomes),
                target_source=target_source,
                terminal_reached=any(outcome.terminal_reached for outcome in outcomes),
                censored=censored,
                censor_reason=censor_reason,
            )
        )
    return targets


def _branch_fault_censor_reason(fault: BranchFaultTrace) -> str:
    fault_kind = fault.fault_kind if fault.fault_kind else fault.status
    return f"branch_fault:{fault_kind}"


def _descendant_nodes(
    node_id: str,
    *,
    children: Mapping[str, Sequence[str]],
    resolved_by_id: Mapping[str, ResolvedNodeTrace],
) -> list[ResolvedNodeTrace]:
    result: list[ResolvedNodeTrace] = []
    queue = deque(children.get(node_id, ()))
    seen: set[str] = set()
    while queue:
        child_id = queue.popleft()
        if child_id in seen:
            continue
        seen.add(child_id)
        node = resolved_by_id.get(child_id)
        if node is None:
            continue
        result.append(node)
        queue.extend(children.get(child_id, ()))
    return result


def _leaf_fresh_nodes(
    nodes: Sequence[ResolvedNodeTrace],
    *,
    children: Mapping[str, Sequence[str]],
) -> list[ResolvedNodeTrace]:
    return [
        node
        for node in nodes
        if node.state_score_is_fresh and not children.get(node.node_id)
    ]


def _state_score_top_k_ids(prune: StablePruneTrace, *, k: int) -> set[str]:
    ranked = sorted(prune.nodes, key=lambda node: node.state_score, reverse=True)
    return {node.node_id for node in ranked[:k]}


def _available_trace_actions(actions: Sequence[JsonObject]) -> list[JsonObject]:
    return [
        action
        for action in actions
        if action.get("is_available") is not False
        and isinstance(action.get("action_id"), str)
        and bool(action.get("action_id"))
    ]


def _available_dto_actions(dto: Mapping[str, Any]) -> list[JsonObject]:
    actions = dto.get("legal_actions")
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
        return []
    return [
        action
        for action in actions
        if isinstance(action, Mapping)
        and action.get("is_available") is not False
        and isinstance(action.get("action_id"), str)
        and bool(action.get("action_id"))
    ]


def _effective_time_budget_ms(
    timeout_s: float,
    configured_time_budget_ms: float | None,
) -> float:
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(float(timeout_s))
        or timeout_s <= 0
    ):
        raise ValueError("timeout_s must be a finite positive number")
    timeout_ms = float(timeout_s) * 1000.0
    if configured_time_budget_ms is None:
        return timeout_ms
    return min(timeout_ms, float(configured_time_budget_ms))


def _independent_policy_copy(policy: PolicyModel) -> PolicyModel:
    try:
        copied = copy.deepcopy(policy)
    except Exception as exc:  # noqa: BLE001 - fail closed rather than sharing runtime state
        raise TypeError(
            "BudgetedOracleCollector.from_beam_engine requires a deepcopy-able policy so "
            "teacher inference cannot mutate runtime policy state; construct the collector "
            "explicitly with an independent policy instance instead"
        ) from exc
    if copied is policy:
        raise TypeError(
            "policy deepcopy returned the runtime policy object; construct the oracle "
            "explicitly with an independent policy instance"
        )
    return copied


def _independent_value_copy(value_fn: ValueModel) -> ValueModel:
    try:
        copied = copy.deepcopy(value_fn)
    except Exception as exc:  # noqa: BLE001 - fail closed rather than sharing runtime state
        raise TypeError(
            "BudgetedOracleCollector.from_beam_engine requires a deepcopy-able ValueModel "
            "so teacher inference cannot mutate runtime value state; construct the "
            "collector explicitly with an independent ValueModel instance instead"
        ) from exc
    if copied is value_fn:
        raise TypeError(
            "ValueModel deepcopy returned the runtime ValueModel object; construct the "
            "oracle explicitly with an independent ValueModel instance"
        )
    return copied


def _oracle_provenance(policy: PolicyModel, value_fn: ValueModel) -> OracleProvenance:
    policy_class = _qualified_class_name(policy)
    policy_metadata = _model_provenance_metadata(policy, "PolicyModel")
    if isinstance(policy, CoverageConstrainedPolicy):
        inner_policy_class = _qualified_class_name(policy.inner)
        inner_policy_metadata = _model_provenance_metadata(
            policy.inner, "inner PolicyModel"
        )
        coverage_policy_class = policy_class
    else:
        inner_policy_class = policy_class
        inner_policy_metadata = policy_metadata
        coverage_policy_class = None
    return OracleProvenance(
        teacher_policy_class=policy_class,
        teacher_inner_policy_class=inner_policy_class,
        teacher_coverage_policy_class=coverage_policy_class,
        teacher_value_class=_qualified_class_name(value_fn),
        teacher_policy_metadata=policy_metadata,
        teacher_inner_policy_metadata=inner_policy_metadata,
        teacher_value_metadata=_model_provenance_metadata(value_fn, "ValueModel"),
    )


def _model_provenance_metadata(value: Any, label: str) -> JsonObject:
    hook = getattr(value, "oracle_provenance", None)
    if not callable(hook):
        return {}
    raw = hook()
    if not isinstance(raw, Mapping):
        raise TypeError(f"{label}.oracle_provenance must return a mapping")
    try:
        encoded = json.dumps(raw, ensure_ascii=False, sort_keys=True, allow_nan=False)
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{label}.oracle_provenance must return JSON-serializable finite metadata"
        ) from exc
    if not isinstance(normalized, dict):
        raise TypeError(f"{label}.oracle_provenance must serialize to a JSON object")
    return normalized


def _qualified_class_name(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"
