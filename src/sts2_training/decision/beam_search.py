"""Policy-guided Combat beam search over `AsyncTrainingApiClient`.

Ordinary combat actions consume `max_depth`. Interactive choice continuations are
resolved as local macro-action steps: continuation DTOs are expanded using policy order
without passing those intermediate states through the global ValueModel. Stable/terminal
states receive a ``state_score`` only after the continuation resolves. This keeps
ValueModel's meaning centered on resolved Combat states while still bounding continuation
work with `max_continuation_steps` and `beam_width`.

Policy ranking produces a cheap pre-simulation ``action_score``. Stable/resolved frontier
selection is delegated to ``StableFrontierPruner``, whose learned implementation may
produce a separate ``frontier_score``. Historical dataclass fields and timing fields keep
their existing spellings for compatibility; runtime code uses canonical score accessors.

Policy candidate limits, continuation ordering, and Whole Run active-branch capacity
remain separate mechanisms owned by this engine.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from sts2_training.api.contract import ROOT_BRANCH_ID, RequestRejectedError
from sts2_training.api.transport import TransportError
from sts2_training.decision.candidate_coverage import (
    CandidateProposal,
    propose_batch_with_provenance,
)
from sts2_training.decision.combat_decision import (
    CONTINUATION_ACTION_TYPES,
    action_type_for_id,
    available_action_types,
    is_continuation_decision,
)
from sts2_training.decision.policy import PolicyModel
from sts2_training.decision.search_trace import (
    BranchFaultTrace,
    PolicyCandidateTrace,
    PolicyProposalTrace,
    SearchTraceCollector,
    SearchTraceEnd,
    SearchTraceEvent,
    SearchTraceStart,
    StablePruneNodeTrace,
    StablePruneTrace,
)
from sts2_training.decision.stable_pruner import (
    StableFrontierPruner,
    StablePruneContext,
    StablePruneNodeView,
    ValueTopKPruner,
)
from sts2_training.decision.value import ValueModel

JsonObject = dict[str, Any]
_LOG = logging.getLogger(__name__)
_RESOLVED_STATUSES = frozenset({"completed", "partial"})
_WHOLE_RUN_COMBAT_BOUNDARIES = frozenset({"stable", "pending_choice"})


@dataclass
class BeamSearchConfig:
    beam_width: int = 8
    top_k_actions: int = 4
    max_depth: int = 2
    simulation_options: Mapping[str, Any] | None = None
    time_budget_ms: float | None = None
    max_batch_size: int = 64
    expand_partial: bool = True
    release_branches_on_finish: bool = True
    beam_searchable_action_types: frozenset[str] = field(
        default_factory=lambda: frozenset({"system", "card", "potion"})
    )
    max_continuation_steps: int = 8

    def __post_init__(self) -> None:
        for name in (
            "beam_width",
            "top_k_actions",
            "max_depth",
            "max_batch_size",
            "max_continuation_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.time_budget_ms is not None and (
            isinstance(self.time_budget_ms, bool)
            or not isinstance(self.time_budget_ms, (int, float))
            or not math.isfinite(float(self.time_budget_ms))
            or self.time_budget_ms <= 0
        ):
            raise ValueError("time_budget_ms must be a finite positive number when provided")
        if self.simulation_options is None:
            self.simulation_options = {"stop_condition": "next_decision"}


@dataclass(frozen=True)
class BeamNode:
    branch_id: str
    parent_branch_id: str
    rng_id: int
    decision_point_id: str
    masked_emulator_dto: Mapping[str, Any]
    depth: int
    value: float
    root_action_id: str | None
    combat_depth: int = 0
    continuation_steps: int = 0
    branch_log: tuple[Any, ...] = ()
    terminal: bool = False
    action_id: str | None = None
    action_type: str | None = None
    action: Mapping[str, Any] | None = None
    policy_rank: int | None = None
    policy_score: float | None = None
    post_coverage_rank: int | None = None
    candidate_source: str | None = None

    @property
    def state_score(self) -> float:
        """Canonical name for the resolved state's ValueModel score."""

        return self.value

    @property
    def action_rank(self) -> int | None:
        """Canonical name for the action's pre-coverage rank."""

        return self.policy_rank

    @property
    def action_score(self) -> float | None:
        """Canonical name for the action's pre-simulation score."""

        return self.policy_score


@dataclass
class BeamSearchStats:
    depths_completed: int = 0
    nodes_expanded: int = 0
    branches_created: int = 0
    branches_faulted: int = 0
    policy_ms: float = 0.0
    emulate_actions_ms: float = 0.0
    value_ms: float = 0.0
    cleanup_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def action_score_ms(self) -> float:
        """Compatibility view of time spent producing/ranking action candidates."""

        return self.policy_ms

    @property
    def state_score_ms(self) -> float:
        """Compatibility view of time spent evaluating resolved state scores."""

        return self.value_ms


@dataclass(frozen=True)
class BeamSearchResult:
    best_root_action_id: str | None
    best_value: float | None
    best_node: BeamNode | None
    reason: str
    stats: BeamSearchStats

    @property
    def best_node_score(self) -> float | None:
        """Canonical name for the score attributed to the selected search-tree node."""

        return self.best_value


class BranchIdAllocator:
    def __init__(self, prefix: str | None = None) -> None:
        self._prefix = prefix or uuid.uuid4().hex[:8]
        self._branch_counter = 0
        self._rng_counter = 0

    def next_branch_id(self) -> str:
        self._branch_counter += 1
        return f"bs-{self._prefix}-{self._branch_counter}"

    def next_rng_id(self) -> int:
        self._rng_counter += 1
        return self._rng_counter


class BeamSearchEngine:
    """Construct once per API client/instance so branch/rng IDs stay unique."""

    def __init__(
        self,
        client: Any,
        *,
        policy: PolicyModel,
        value_fn: ValueModel,
        config: BeamSearchConfig | None = None,
        stable_pruner: StableFrontierPruner | None = None,
        trace_collector: SearchTraceCollector | None = None,
    ) -> None:
        self._client = client
        self._policy = policy
        self._value_fn = value_fn
        self.config = config or BeamSearchConfig()
        self._stable_pruner = stable_pruner or ValueTopKPruner()
        self._trace_collector = trace_collector
        self._allocator = BranchIdAllocator()

    @property
    def stable_pruner(self) -> StableFrontierPruner:
        return self._stable_pruner

    @stable_pruner.setter
    def stable_pruner(self, pruner: StableFrontierPruner) -> None:
        if not isinstance(pruner, StableFrontierPruner):
            raise TypeError("stable_pruner must be a StableFrontierPruner")
        self._stable_pruner = pruner

    @property
    def trace_collector(self) -> SearchTraceCollector | None:
        return self._trace_collector

    @trace_collector.setter
    def trace_collector(self, collector: SearchTraceCollector | None) -> None:
        self._trace_collector = collector

    async def search(
        self,
        instance_id: str,
        root_decision: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> BeamSearchResult:
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(float(timeout_s))
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be a finite positive number")

        cfg = self.config
        overall_deadline = time.monotonic() + timeout_s
        search_deadline = overall_deadline
        if cfg.time_budget_ms is not None:
            search_deadline = min(
                search_deadline, time.monotonic() + cfg.time_budget_ms / 1000.0
            )

        stats = BeamSearchStats()
        t_start = time.monotonic()
        root_dto = root_decision.get("masked_emulator_dto")
        root_decision_point_id = root_decision.get("decision_point_id")
        if not isinstance(root_dto, Mapping) or not isinstance(root_decision_point_id, str):
            return BeamSearchResult(None, None, None, "invalid_root_decision", stats)
        if not root_dto.get("legal_actions"):
            return BeamSearchResult(None, None, None, "no_legal_actions", stats)
        if (
            getattr(self._client, "instance_type", None) == "whole_run"
            and _server_batch_limit(self._client) is None
        ):
            return BeamSearchResult(None, None, None, "emulate_actions_not_supported", stats)
        if not _is_beam_searchable_for_client(
            self._client, root_dto, cfg.beam_searchable_action_types
        ):
            return BeamSearchResult(None, None, None, "not_beam_searchable", stats)

        search_id = uuid.uuid4().hex
        self._record_trace(
            SearchTraceStart(
                search_id=search_id,
                instance_id=instance_id,
                root_decision_point_id=root_decision_point_id,
                beam_width=cfg.beam_width,
                top_k_actions=cfg.top_k_actions,
                max_depth=cfg.max_depth,
                max_continuation_steps=cfg.max_continuation_steps,
                time_budget_ms=(
                    None if cfg.time_budget_ms is None else float(cfg.time_budget_ms)
                ),
                pruner_name=self._stable_pruner.name,
                pruner_version=self._stable_pruner.version,
            )
        )

        beam: list[BeamNode] = [
            BeamNode(
                branch_id=ROOT_BRANCH_ID,
                parent_branch_id=ROOT_BRANCH_ID,
                rng_id=0,
                decision_point_id=root_decision_point_id,
                masked_emulator_dto=root_dto,
                depth=0,
                value=0.0,
                root_action_id=None,
            )
        ]
        waiting_stable: list[BeamNode] = []
        finished: list[BeamNode] = []
        all_branch_ids: list[str] = []
        released_branch_ids: set[str] = set()
        reason = "max_depth"
        search_error: BaseException | None = None
        proposal_step_index = 0
        prune_step_index = 0

        try:
            while beam:
                if time.monotonic() >= search_deadline:
                    reason = "time_budget"
                    break

                searchable: list[BeamNode] = []
                for node in beam:
                    if _is_beam_searchable_for_client(
                        self._client,
                        node.masked_emulator_dto,
                        cfg.beam_searchable_action_types,
                    ):
                        searchable.append(node)
                    elif _is_macro_resolved(node):
                        finished.append(node)
                beam = searchable
                if not beam:
                    reason = "not_beam_searchable"
                    break

                items, item_meta, action_score_ms = self._propose_frontier(
                    beam,
                    search_id=search_id,
                    proposal_step_index=proposal_step_index,
                )
                proposal_step_index += 1
                stats.policy_ms += action_score_ms
                if not items:
                    reason = "no_candidates"
                    break

                (
                    beam,
                    waiting_stable,
                    items,
                    item_meta,
                    capacity_finished,
                    capacity_limited,
                ) = self._fit_whole_run_frontier_to_active_capacity(
                    beam,
                    waiting_stable,
                    items,
                    item_meta,
                )
                finished.extend(capacity_finished)
                if capacity_limited:
                    released = await self._release_unneeded_whole_run_branches(
                        instance_id,
                        all_branch_ids,
                        released_branch_ids,
                        beam,
                        waiting_stable,
                        deadline=search_deadline,
                    )
                    if not released:
                        reason = "active_branch_cleanup_failed"
                        break
                if not items:
                    reason = "active_branch_capacity"
                    break

                branch_results, emulated_item_meta, fatal_reason = await self._emulate_depth_batch(
                    instance_id, items, item_meta, all_branch_ids, stats, search_deadline
                )
                (
                    next_beam,
                    newly_finished,
                    state_score_ms,
                    hit_max_depth,
                    hit_continuation_limit,
                    branches_faulted,
                ) = self._score_frontier(
                    emulated_item_meta, branch_results, search_id=search_id
                )
                stats.value_ms += state_score_ms
                stats.branches_faulted += branches_faulted
                stats.nodes_expanded += len(branch_results)
                if (
                    emulated_item_meta
                    and not next_beam
                    and not newly_finished
                    and not hit_continuation_limit
                ):
                    if _has_unresolved_out_of_scope_result(
                        self._client,
                        branch_results,
                        cfg.beam_searchable_action_types,
                    ):
                        reason = "not_beam_searchable"
                        break
                    raise RuntimeError("all emulate_actions branch results faulted")
                finished.extend(newly_finished)

                continuation_nodes = [
                    node
                    for node in next_beam
                    if is_continuation_decision(node.masked_emulator_dto)
                ]
                stable_nodes = [
                    node
                    for node in next_beam
                    if not is_continuation_decision(node.masked_emulator_dto)
                ]

                if continuation_nodes:
                    waiting_stable.extend(stable_nodes)
                    beam = continuation_nodes[: cfg.beam_width]
                else:
                    stable_nodes.extend(waiting_stable)
                    waiting_stable = []
                    beam = self._prune_stable_frontier(
                        stable_nodes,
                        search_id=search_id,
                        prune_step_index=prune_step_index,
                        phase="stable_frontier",
                        search_deadline=search_deadline,
                        depths_completed=stats.depths_completed,
                    )
                    prune_step_index += 1

                if fatal_reason is not None:
                    reason = fatal_reason
                    break
                stats.depths_completed += 1

                if not beam:
                    if waiting_stable:
                        beam = self._prune_stable_frontier(
                            waiting_stable,
                            search_id=search_id,
                            prune_step_index=prune_step_index,
                            phase="waiting_stable_fallback",
                            search_deadline=search_deadline,
                            depths_completed=stats.depths_completed,
                        )
                        prune_step_index += 1
                        waiting_stable = []
                    elif hit_continuation_limit:
                        reason = "max_continuation_steps"
                        break
                    elif hit_max_depth:
                        reason = "max_depth"
                        break
                    else:
                        reason = "beam_exhausted"
                        break

                released = await self._release_unneeded_whole_run_branches(
                    instance_id,
                    all_branch_ids,
                    released_branch_ids,
                    beam,
                    waiting_stable,
                    deadline=search_deadline,
                )
                if not released:
                    reason = "active_branch_cleanup_failed"
                    break

            finished.extend(waiting_stable)
            finished.extend(node for node in beam if _is_macro_resolved(node))
        except BaseException as exc:
            search_error = exc
            raise
        finally:
            t0 = time.monotonic()
            try:
                await self._cleanup(
                    instance_id,
                    [bid for bid in all_branch_ids if bid not in released_branch_ids],
                    deadline=overall_deadline,
                )
            except Exception:
                if search_error is None:
                    raise
                _LOG.exception(
                    "beam search cleanup also failed while propagating a search error "
                    "for instance_id=%s",
                    instance_id,
                )
            finally:
                stats.cleanup_ms += (time.monotonic() - t0) * 1000.0

        stats.total_ms = (time.monotonic() - t_start) * 1000.0
        actionable = [
            node
            for node in finished
            if node.root_action_id is not None and _is_macro_resolved(node)
        ]
        if not actionable:
            result = BeamSearchResult(None, None, None, reason, stats)
        else:
            best_node = max(actionable, key=lambda node: node.state_score)
            result = BeamSearchResult(
                best_node.root_action_id,
                best_node.state_score,
                best_node,
                reason,
                stats,
            )
        self._record_trace(
            SearchTraceEnd(
                search_id=search_id,
                reason=result.reason,
                best_root_action_id=result.best_root_action_id,
                best_value=result.best_node_score,
                depths_completed=result.stats.depths_completed,
                nodes_expanded=result.stats.nodes_expanded,
                branches_created=result.stats.branches_created,
                branches_faulted=result.stats.branches_faulted,
            )
        )
        return result

    def _propose_frontier(
        self,
        beam: Sequence[BeamNode],
        *,
        search_id: str,
        proposal_step_index: int,
    ) -> tuple[
        list[JsonObject],
        list[tuple[BeamNode, CandidateProposal, str, int]],
        float,
    ]:
        t0 = time.monotonic()
        requests = [
            (node.masked_emulator_dto.get("legal_actions") or [], node.masked_emulator_dto)
            for node in beam
        ]
        proposals = propose_batch_with_provenance(
            self._policy,
            requests,
            top_k=self.config.top_k_actions,
        )
        action_score_ms = (time.monotonic() - t0) * 1000.0
        if len(proposals) != len(beam):
            raise RuntimeError("PolicyModel.propose_batch must return exactly one entry per request")

        items: list[JsonObject] = []
        item_meta: list[tuple[BeamNode, CandidateProposal, str, int]] = []
        for parent_index, (node, candidates) in enumerate(zip(beam, proposals)):
            if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
                raise RuntimeError("PolicyModel.propose_batch entries must be candidate sequences")
            if len(candidates) > self.config.top_k_actions:
                raise RuntimeError("PolicyModel.propose_batch returned more than top_k candidates")
            available_ids = _available_action_ids(node.masked_emulator_dto)
            trace_candidates: list[PolicyCandidateTrace] = []
            for candidate in candidates:
                if not isinstance(candidate, CandidateProposal):
                    raise RuntimeError("policy provenance adapter returned an invalid candidate")
                if candidate.action_id not in available_ids:
                    raise RuntimeError(
                        "PolicyModel proposed an action_id that is not currently available: "
                        f"{candidate.action_id!r}"
                    )
                branch_id = self._allocator.next_branch_id()
                rng_id = (
                    node.rng_id
                    if node.branch_id != ROOT_BRANCH_ID
                    else self._allocator.next_rng_id()
                )
                items.append(
                    {
                        "parent_branch_id": node.branch_id,
                        "branch_id": branch_id,
                        "rng_id": rng_id,
                        "decision_point_id": node.decision_point_id,
                        "action_id": candidate.action_id,
                    }
                )
                item_meta.append((node, candidate, branch_id, rng_id))
                if self._trace_collector is not None:
                    action = _action_for_id(node.masked_emulator_dto, candidate.action_id)
                    trace_candidates.append(
                        PolicyCandidateTrace(
                            action_id=candidate.action_id,
                            action={} if action is None else dict(action),
                            branch_id=branch_id,
                            rng_id=rng_id,
                            policy_rank=candidate.action_rank,
                            policy_score=candidate.action_score,
                            post_coverage_rank=candidate.post_coverage_rank,
                            candidate_source=candidate.candidate_source,
                        )
                    )

            if self._trace_collector is not None:
                self._record_trace(
                    PolicyProposalTrace(
                        search_id=search_id,
                        proposal_step_id=(
                            f"{search_id}:proposal:{proposal_step_index}:{parent_index}"
                        ),
                        parent_node_id=_trace_node_id(search_id, node.branch_id),
                        parent_branch_id=node.branch_id,
                        decision_point_id=node.decision_point_id,
                        legal_actions=tuple(
                            dict(action)
                            for action in _legal_action_mappings(node.masked_emulator_dto)
                        ),
                        candidates=tuple(trace_candidates),
                    )
                )
        return items, item_meta, action_score_ms

    def _prune_stable_frontier(
        self,
        frontier: Sequence[BeamNode],
        *,
        search_id: str,
        prune_step_index: int,
        phase: str,
        search_deadline: float,
        depths_completed: int,
    ) -> list[BeamNode]:
        prune_step_id = f"{search_id}:prune:{prune_step_index}"
        remaining_time_ms = max(0.0, (search_deadline - time.monotonic()) * 1000.0)
        context = StablePruneContext(
            search_id=search_id,
            prune_step_id=prune_step_id,
            phase=phase,
            beam_width=self.config.beam_width,
            max_depth=self.config.max_depth,
            depths_completed=depths_completed,
            remaining_time_ms=remaining_time_ms,
        )
        frontier_views = tuple(_stable_prune_node_view(node) for node in frontier)
        selected_indices = self._stable_pruner.select(
            frontier_views,
            k=self.config.beam_width,
            context=context,
        )
        selected_indices = _validated_pruner_indices(
            selected_indices,
            frontier_size=len(frontier),
            k=self.config.beam_width,
        )
        selected = [frontier[index] for index in selected_indices]

        if self._trace_collector is not None:
            selected_index_set = set(selected_indices)
            self._record_trace(
                StablePruneTrace(
                    search_id=search_id,
                    prune_step_id=prune_step_id,
                    phase=phase,
                    k=self.config.beam_width,
                    frontier_size=len(frontier),
                    pruner_name=self._stable_pruner.name,
                    pruner_version=self._stable_pruner.version,
                    max_depth=self.config.max_depth,
                    depths_completed=depths_completed,
                    remaining_time_ms=remaining_time_ms,
                    nodes=tuple(
                        StablePruneNodeTrace(
                            node_id=_trace_node_id(search_id, node.branch_id),
                            parent_node_id=_trace_node_id(
                                search_id, node.parent_branch_id
                            ),
                            branch_id=node.branch_id,
                            parent_branch_id=node.parent_branch_id,
                            frontier_index_before_prune=index,
                            kept=index in selected_index_set,
                            value=view.state_score,
                            root_action_id=view.root_action_id,
                            rng_id=node.rng_id,
                            decision_point_id=node.decision_point_id,
                            depth=view.depth,
                            combat_depth=view.combat_depth,
                            continuation_steps=view.continuation_steps,
                            terminal=view.terminal,
                            action_id=node.action_id,
                            action_type=view.action_type,
                            action=None if node.action is None else dict(node.action),
                            policy_rank=view.action_rank,
                            policy_score=view.action_score,
                            post_coverage_rank=view.post_coverage_rank,
                            candidate_source=view.candidate_source,
                        )
                        for index, (node, view) in enumerate(zip(frontier, frontier_views))
                    ),
                    selected_indices=tuple(selected_indices),
                )
            )
        return selected

    def _record_trace(self, event: SearchTraceEvent) -> None:
        if self._trace_collector is not None:
            self._trace_collector.record(event)

    def _fit_whole_run_frontier_to_active_capacity(
        self,
        beam: Sequence[BeamNode],
        waiting_stable: Sequence[BeamNode],
        items: Sequence[JsonObject],
        item_meta: Sequence[tuple[BeamNode, CandidateProposal, str, int]],
    ) -> tuple[
        list[BeamNode],
        list[BeamNode],
        list[JsonObject],
        list[tuple[BeamNode, CandidateProposal, str, int]],
        list[BeamNode],
        bool,
    ]:
        """Fit a Whole Run frontier inside RL's shared active Branch budget."""
        capacity = _server_active_branch_capacity(self._client)
        if capacity is None:
            return list(beam), list(waiting_stable), list(items), list(item_meta), [], False

        indices_by_parent: dict[str, list[int]] = {}
        for index, meta in enumerate(item_meta):
            indices_by_parent.setdefault(meta[0].branch_id, []).append(index)

        selected_beam: list[BeamNode] = []
        selected_indices: list[int] = []
        used_slots = 0
        limited = False

        for node in beam:
            candidate_indices = indices_by_parent.get(node.branch_id, [])
            if not candidate_indices:
                limited = True
                continue
            parent_cost = 0 if node.branch_id == ROOT_BRANCH_ID else 1
            remaining_for_children = capacity - used_slots - parent_cost
            if remaining_for_children <= 0:
                limited = True
                break
            take = min(len(candidate_indices), remaining_for_children)
            selected_beam.append(node)
            selected_indices.extend(candidate_indices[:take])
            used_slots += parent_cost + take
            if take < len(candidate_indices):
                limited = True
                break

        selected_parent_ids = {node.branch_id for node in selected_beam}
        pruned_beam = [node for node in beam if node.branch_id not in selected_parent_ids]
        if pruned_beam:
            limited = True

        waiting_budget = max(0, capacity - used_slots)
        ranked_waiting = sorted(
            waiting_stable,
            key=lambda node: node.state_score,
            reverse=True,
        )
        kept_waiting = ranked_waiting[:waiting_budget]
        pruned_waiting = ranked_waiting[waiting_budget:]
        if pruned_waiting:
            limited = True

        capacity_finished = [
            node for node in (*pruned_beam, *pruned_waiting) if _is_macro_resolved(node)
        ]

        selected_items = [items[index] for index in selected_indices]
        selected_meta = [item_meta[index] for index in selected_indices]
        if len(selected_items) != len(items):
            limited = True

        return (
            selected_beam,
            kept_waiting,
            selected_items,
            selected_meta,
            capacity_finished,
            limited,
        )

    async def _emulate_depth_batch(
        self,
        instance_id: str,
        items: Sequence[JsonObject],
        item_meta: Sequence[tuple[BeamNode, CandidateProposal, str, int]],
        all_branch_ids: list[str],
        stats: BeamSearchStats,
        deadline: float,
    ) -> tuple[
        dict[str, Any],
        list[tuple[BeamNode, CandidateProposal, str, int]],
        str | None,
    ]:
        cfg = self.config
        batch_size = cfg.max_batch_size
        server_limit = _server_batch_limit(self._client)
        if server_limit is not None:
            batch_size = min(batch_size, server_limit)

        branch_results: dict[str, Any] = {}
        emulated_item_meta: list[tuple[BeamNode, CandidateProposal, str, int]] = []
        for start in range(0, len(items), batch_size):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return branch_results, emulated_item_meta, "time_budget"
            chunk_items = items[start : start + batch_size]
            chunk_meta = item_meta[start : start + batch_size]
            t0 = time.monotonic()
            try:
                response = await self._client.emulate_actions(
                    instance_id,
                    chunk_items,
                    timeout_s=remaining,
                    simulation_options=cfg.simulation_options,
                )
            except RequestRejectedError as exc:
                stats.emulate_actions_ms += (time.monotonic() - t0) * 1000.0
                if _client_unusable(self._client):
                    raise
                fault_kind = exc.response.get("fault_kind")
                detail = (
                    fault_kind
                    if isinstance(fault_kind, str) and fault_kind
                    else type(exc).__name__
                )
                return (
                    branch_results,
                    emulated_item_meta,
                    f"emulate_actions_rejected:{detail}",
                )
            stats.emulate_actions_ms += (time.monotonic() - t0) * 1000.0
            all_branch_ids.extend(meta[2] for meta in chunk_meta)
            stats.branches_created += len(chunk_meta)
            emulated_item_meta.extend(chunk_meta)
            branch_results.update(response.get("branch_results") or {})
        return branch_results, emulated_item_meta, None

    def _score_frontier(
        self,
        item_meta: Sequence[tuple[BeamNode, Any, str, int]],
        branch_results: Mapping[str, Any],
        depth: int | None = None,
        *,
        search_id: str,
    ) -> tuple[list[BeamNode], list[BeamNode], float, bool, bool, int]:
        del depth
        cfg = self.config
        resolved: list[
            tuple[
                BeamNode,
                Any,
                str,
                int,
                Mapping[str, Any],
                Mapping[str, Any],
                Any,
            ]
        ] = []
        branches_faulted = 0
        for node, candidate, branch_id, rng_id in item_meta:
            result = branch_results.get(branch_id)
            if not isinstance(result, Mapping) or result.get("status") not in _RESOLVED_STATUSES:
                self._record_branch_fault(
                    search_id,
                    node,
                    candidate,
                    branch_id,
                    rng_id,
                    status=result.get("status") if isinstance(result, Mapping) else None,
                    fault_kind=result.get("fault_kind") if isinstance(result, Mapping) else None,
                    detail=result.get("error") if isinstance(result, Mapping) else None,
                )
                branches_faulted += 1
                continue
            dto = result.get("masked_emulator_dto")
            if isinstance(dto, Mapping):
                resolved.append(
                    (node, candidate, branch_id, rng_id, result, dto, result.get("status"))
                )
            else:
                self._record_branch_fault(
                    search_id,
                    node,
                    candidate,
                    branch_id,
                    rng_id,
                    status="invalid_masked_emulator_dto",
                    fault_kind=None,
                    detail=None,
                )
                branches_faulted += 1

        state_score_entries = [
            entry
            for entry in resolved
            if _is_terminal(entry[5])
            or (
                not is_continuation_decision(entry[5])
                and not _is_whole_run_unresolved_out_of_scope(
                    self._client,
                    entry[5],
                    cfg.beam_searchable_action_types,
                )
            )
        ]
        t0 = time.monotonic()
        raw_state_scores = (
            self._value_fn.evaluate_batch([entry[5] for entry in state_score_entries])
            if state_score_entries
            else []
        )
        state_score_ms = (time.monotonic() - t0) * 1000.0
        state_scores = _validated_state_scores(
            raw_state_scores,
            expected=len(state_score_entries),
        )
        state_score_by_branch = {
            entry[2]: state_score
            for entry, state_score in zip(state_score_entries, state_scores)
        }

        next_beam: list[BeamNode] = []
        newly_finished: list[BeamNode] = []
        hit_max_depth = False
        hit_continuation_limit = False

        for node, candidate, branch_id, rng_id, result, dto, status in resolved:
            action_type = action_type_for_id(node.masked_emulator_dto, candidate.action_id)
            if action_type is None:
                raise RuntimeError(
                    f"selected action_id {candidate.action_id!r} has no valid action_type"
                )
            action = _action_for_id(node.masked_emulator_dto, candidate.action_id)
            is_continuation_action = action_type in CONTINUATION_ACTION_TYPES
            combat_depth = node.combat_depth + (0 if is_continuation_action else 1)
            continuation_steps = node.continuation_steps + 1 if is_continuation_action else 0
            terminal = _is_terminal(dto)
            if _is_whole_run_unresolved_out_of_scope(
                self._client,
                dto,
                cfg.beam_searchable_action_types,
            ):
                continue
            state_score = state_score_by_branch.get(branch_id, node.state_score)
            decision_point_id = result.get("decision_point_id")
            new_node = BeamNode(
                branch_id=branch_id,
                parent_branch_id=node.branch_id,
                rng_id=rng_id,
                decision_point_id=(
                    decision_point_id if isinstance(decision_point_id, str) else ""
                ),
                masked_emulator_dto=dto,
                depth=node.depth + 1,
                value=state_score,
                root_action_id=node.root_action_id or candidate.action_id,
                combat_depth=combat_depth,
                continuation_steps=continuation_steps,
                branch_log=tuple(result.get("branch_log") or ()),
                terminal=terminal,
                action_id=candidate.action_id,
                action_type=action_type,
                action=None if action is None else dict(action),
                policy_rank=candidate.action_rank,
                policy_score=candidate.action_score,
                post_coverage_rank=candidate.post_coverage_rank,
                candidate_source=candidate.candidate_source,
            )
            cannot_expand = not new_node.decision_point_id or (
                status == "partial" and not cfg.expand_partial
            )
            if terminal or cannot_expand:
                newly_finished.append(new_node)
                continue
            if is_continuation_decision(dto):
                if continuation_steps >= cfg.max_continuation_steps:
                    hit_continuation_limit = True
                    continue
                next_beam.append(new_node)
                continue
            if combat_depth >= cfg.max_depth:
                hit_max_depth = True
                newly_finished.append(new_node)
            else:
                next_beam.append(new_node)

        return (
            next_beam,
            newly_finished,
            state_score_ms,
            hit_max_depth,
            hit_continuation_limit,
            branches_faulted,
        )

    def _record_branch_fault(
        self,
        search_id: str,
        node: BeamNode,
        candidate: Any,
        branch_id: str,
        rng_id: int,
        *,
        status: str | None,
        fault_kind: str | None,
        detail: str | None,
    ) -> None:
        """Log and trace a branch whose `emulate_actions` result never resolved.

        Called from the exact spots that would otherwise `continue` past the fault with
        no record at all - the only way this class of failure stays invisible when some
        other branch in the same frontier happens to succeed.
        """

        status_label = status if isinstance(status, str) and status else "missing_result"
        action_type = action_type_for_id(node.masked_emulator_dto, candidate.action_id)
        action = _action_for_id(node.masked_emulator_dto, candidate.action_id)
        is_continuation_action = action_type in CONTINUATION_ACTION_TYPES
        combat_depth = node.combat_depth + (0 if is_continuation_action else 1)
        continuation_steps = node.continuation_steps + 1 if is_continuation_action else 0
        log_level = logging.INFO if fault_kind == "validation_rejection" else logging.WARNING
        _LOG.log(
            log_level,
            "beam search branch %s (parent=%s, action_id=%s, action_type=%s) did not "
            "resolve: status=%s fault_kind=%s detail=%s",
            branch_id,
            node.branch_id,
            candidate.action_id,
            action_type,
            status_label,
            fault_kind,
            detail,
        )
        self._record_trace(
            BranchFaultTrace(
                search_id=search_id,
                node_id=f"{search_id}:{branch_id}",
                parent_node_id=f"{search_id}:{node.branch_id}",
                branch_id=branch_id,
                parent_branch_id=node.branch_id,
                root_action_id=node.root_action_id or candidate.action_id,
                rng_id=rng_id,
                depth=node.depth + 1,
                combat_depth=combat_depth,
                continuation_steps=continuation_steps,
                action_id=candidate.action_id,
                action_type=action_type,
                action=None if action is None else dict(action),
                policy_rank=candidate.action_rank,
                policy_score=candidate.action_score,
                post_coverage_rank=candidate.post_coverage_rank,
                candidate_source=candidate.candidate_source,
                status=status_label,
                fault_kind=fault_kind,
                detail=detail,
            )
        )

    async def _release_unneeded_whole_run_branches(
        self,
        instance_id: str,
        all_branch_ids: list[str],
        released_branch_ids: set[str],
        beam: Sequence[BeamNode],
        waiting_stable: Sequence[BeamNode],
        *,
        deadline: float,
    ) -> bool:
        if (
            getattr(self._client, "instance_type", None) != "whole_run"
            or not self.config.release_branches_on_finish
        ):
            return True
        keep_ids = {
            node.branch_id
            for node in (*beam, *waiting_stable)
            if node.branch_id != ROOT_BRANCH_ID
        }
        releasable = [
            branch_id
            for branch_id in all_branch_ids
            if branch_id not in keep_ids and branch_id not in released_branch_ids
        ]
        if not releasable:
            return True
        if _client_unusable(self._client):
            return False
        if await self._cleanup_call("release_branches", instance_id, releasable, deadline):
            released_branch_ids.update(releasable)
            return True
        return False

    async def _cleanup(
        self,
        instance_id: str,
        branch_ids: list[str],
        *,
        deadline: float,
    ) -> None:
        if not branch_ids or not self.config.release_branches_on_finish:
            return
        if _client_unusable(self._client):
            _LOG.warning(
                "skipping beam search branch cleanup for instance_id=%s: client has "
                "pending_retry or an invalid session",
                instance_id,
            )
            return
        cancelled = await self._cleanup_call(
            "cancel_branches", instance_id, branch_ids, deadline
        )
        if not cancelled:
            _LOG.warning(
                "skipping beam search branch release for instance_id=%s because "
                "cancellation did not complete",
                instance_id,
            )
            return
        if _client_unusable(self._client):
            _LOG.warning(
                "skipping beam search branch release for instance_id=%s after "
                "cancellation failure",
                instance_id,
            )
            return
        await self._cleanup_call("release_branches", instance_id, branch_ids, deadline)

    async def _cleanup_call(
        self,
        operation: str,
        instance_id: str,
        branch_ids: list[str],
        deadline: float,
    ) -> bool:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _LOG.warning(
                "skipping beam search %s for instance_id=%s: timeout budget exhausted",
                operation,
                instance_id,
            )
            return False
        try:
            await getattr(self._client, operation)(
                instance_id, branch_ids, timeout_s=min(5.0, remaining)
            )
        except RequestRejectedError:
            if _client_unusable(self._client):
                raise
            _LOG.warning(
                "beam search %s was rejected for instance_id=%s (%d branches)",
                operation,
                instance_id,
                len(branch_ids),
                exc_info=True,
            )
            return False
        except TransportError as exc:
            if exc.completion_uncertain or _client_unusable(self._client):
                raise
            _LOG.warning(
                "beam search %s transport failure for instance_id=%s (%d branches)",
                operation,
                instance_id,
                len(branch_ids),
                exc_info=True,
            )
            return False
        return True


def _server_batch_limit(client: Any) -> int | None:
    value = getattr(client, "max_emulate_actions_items", None)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _server_active_branch_capacity(client: Any) -> int | None:
    if getattr(client, "instance_type", None) != "whole_run":
        return None
    return _server_batch_limit(client)


def _legal_action_mappings(dto: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    legal_actions = dto.get("legal_actions")
    if not isinstance(legal_actions, Sequence) or isinstance(legal_actions, (str, bytes)):
        return []
    return [action for action in legal_actions if isinstance(action, Mapping)]


def _available_action_ids(dto: Mapping[str, Any]) -> set[str]:
    return {
        action_id
        for action in _legal_action_mappings(dto)
        if action.get("is_available") is not False
        for action_id in [action.get("action_id")]
        if isinstance(action_id, str) and action_id
    }


def _action_for_id(
    dto: Mapping[str, Any], action_id: str
) -> Mapping[str, Any] | None:
    for action in _legal_action_mappings(dto):
        if action.get("action_id") == action_id:
            return action
    return None


def _stable_prune_node_view(node: BeamNode) -> StablePruneNodeView:
    if is_continuation_decision(node.masked_emulator_dto):
        raise RuntimeError(
            "StableFrontierPruner cannot receive continuation nodes with inherited/stale "
            "state scores"
        )
    return StablePruneNodeView(
        value=node.state_score,
        root_action_id=node.root_action_id,
        depth=node.depth,
        combat_depth=node.combat_depth,
        continuation_steps=node.continuation_steps,
        terminal=node.terminal,
        action_type=node.action_type,
        policy_rank=node.action_rank,
        policy_score=node.action_score,
        post_coverage_rank=node.post_coverage_rank,
        candidate_source=node.candidate_source,
    )


def _validated_pruner_indices(
    selected: object,
    *,
    frontier_size: int,
    k: int,
) -> list[int]:
    if not isinstance(selected, list):
        raise RuntimeError("StableFrontierPruner.select must return list[int]")
    if len(selected) > k:
        raise RuntimeError("StableFrontierPruner.select returned more than k indices")
    seen: set[int] = set()
    normalized: list[int] = []
    for index in selected:
        if isinstance(index, bool) or not isinstance(index, int):
            raise RuntimeError("StableFrontierPruner.select indices must be integers, not bool")
        if index < 0 or index >= frontier_size:
            raise RuntimeError("StableFrontierPruner.select returned an out-of-range index")
        if index in seen:
            raise RuntimeError("StableFrontierPruner.select returned a duplicate index")
        seen.add(index)
        normalized.append(index)
    return normalized


def _trace_node_id(search_id: str, branch_id: str) -> str:
    return f"{search_id}:{branch_id}"


def _validated_state_scores(
    state_scores: Sequence[Any],
    *,
    expected: int,
) -> list[float]:
    if len(state_scores) != expected:
        raise RuntimeError(
            "ValueModel.evaluate_batch must return exactly one state score per dto"
        )
    normalized: list[float] = []
    for raw_state_score in state_scores:
        if isinstance(raw_state_score, (bool, str, bytes)):
            raise RuntimeError(
                "ValueModel.evaluate_batch must return finite numeric state scores"
            )
        try:
            state_score = float(raw_state_score)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                "ValueModel.evaluate_batch must return finite numeric state scores"
            ) from exc
        if not math.isfinite(state_score):
            raise RuntimeError(
                "ValueModel.evaluate_batch must return finite numeric state scores"
            )
        normalized.append(state_score)
    return normalized


def _validated_values(values: Sequence[Any], *, expected: int) -> list[float]:
    """Compatibility alias for callers of the former internal helper name."""

    return _validated_state_scores(values, expected=expected)


def _client_unusable(client: Any) -> bool:
    return getattr(client, "pending_retry", None) is not None or getattr(
        client, "session_invalid", False
    )


def _is_macro_resolved(node: BeamNode) -> bool:
    return node.terminal or not is_continuation_decision(node.masked_emulator_dto)


def _is_beam_searchable_for_client(
    client: Any,
    dto: Mapping[str, Any],
    allowed_action_types: frozenset[str],
) -> bool:
    if (
        getattr(client, "instance_type", None) == "whole_run"
        and dto.get("boundary") not in _WHOLE_RUN_COMBAT_BOUNDARIES
    ):
        return False
    return _is_beam_searchable(dto, allowed_action_types)


def _is_whole_run_unresolved_out_of_scope(
    client: Any,
    dto: Mapping[str, Any],
    allowed_action_types: frozenset[str],
) -> bool:
    return (
        getattr(client, "instance_type", None) == "whole_run"
        and not _is_terminal(dto)
        and not _is_beam_searchable_for_client(client, dto, allowed_action_types)
    )


def _has_unresolved_out_of_scope_result(
    client: Any,
    branch_results: Mapping[str, Any],
    allowed_action_types: frozenset[str],
) -> bool:
    if getattr(client, "instance_type", None) != "whole_run":
        return False
    for result in branch_results.values():
        if not isinstance(result, Mapping) or result.get("status") not in _RESOLVED_STATUSES:
            continue
        dto = result.get("masked_emulator_dto")
        if isinstance(dto, Mapping) and _is_whole_run_unresolved_out_of_scope(
            client,
            dto,
            allowed_action_types,
        ):
            return True
    return False


def _is_beam_searchable(
    dto: Mapping[str, Any], allowed_action_types: frozenset[str]
) -> bool:
    action_types = available_action_types(dto)
    return action_types is not None and bool(action_types) and action_types <= allowed_action_types


def _is_terminal(dto: Mapping[str, Any]) -> bool:
    if dto.get("run_terminal") is True or dto.get("terminal") is True:
        return True
    boundary = dto.get("boundary")
    if isinstance(boundary, str) and boundary in ("terminal", "run_terminal"):
        return True
    transition = dto.get("transition")
    if isinstance(transition, Mapping) and transition.get("kind") == "combat_completed":
        return True
    legal_actions = dto.get("legal_actions")
    if isinstance(legal_actions, Sequence) and not isinstance(legal_actions, (str, bytes)):
        if not any(
            isinstance(action, Mapping) and action.get("is_available") is not False
            for action in legal_actions
        ):
            return True
    return False