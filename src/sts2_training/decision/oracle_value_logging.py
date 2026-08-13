"""Oracle collection extension that persists only root-action ValueModel inputs.

The normal Oracle trace intentionally avoids embedding every branch DTO because doing so
would make training logs grow with the full Beam tree. This module captures only resolved
stable/terminal states at combat depth 1 (the states that ValueModel can actually score
for a root action) and joins one such raw RL DTO to each root-action RNG outcome's deeper
Oracle target.

The raw ``masked_emulator_dto`` is copied without feature normalization. Deeper branch
DTOs remain transient and are never persisted by this contract. A non-terminal root
state's own teacher bootstrap is not a deeper target and is therefore persisted as
``no_target`` rather than becoming a self-imitation label.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from sts2_training.decision.combat_decision import is_continuation_decision
from sts2_training.decision.oracle_search import (
    BudgetedOracleCollector,
    OracleCollectionResult,
    OracleProvenance,
    OracleTargets,
    _OracleBeamSearchEngine,
    _OracleTraceCollector,
    _effective_time_budget_ms,
    _oracle_provenance,
    build_oracle_targets,
)
from sts2_training.decision.search_trace import ResolvedNodeTrace, SearchTraceEvent

JsonObject = Mapping[str, Any]


@dataclass(frozen=True)
class RootActionValueSample:
    """One raw root-action post-state paired with its deeper Oracle value target.

    ``decision_point_id`` identifies the next decision when the post-state is
    non-terminal. Terminal post-states have no next decision and therefore store ``None``.
    """

    action_id: str
    rng_id: int
    root_state_node_id: str
    decision_point_id: str | None
    masked_emulator_dto: JsonObject
    target_value: float | None
    target_source: str
    terminal_reached: bool
    censored: bool
    censor_reason: str | None
    best_node_id: str | None


@dataclass(frozen=True)
class RootValueOracleCollectionResult:
    """Oracle result plus bounded-size ValueModel training samples."""

    search_result: Any
    trace: tuple[SearchTraceEvent, ...]
    targets: OracleTargets
    provenance: OracleProvenance
    root_value_samples: tuple[RootActionValueSample, ...]


class _RootValueCapturingOracleEngine(_OracleBeamSearchEngine):
    """Keep raw DTOs only for resolved root-action states, never the whole Beam tree."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._root_state_dtos: dict[str, dict[str, Any]] = {}

    @property
    def root_state_dtos(self) -> Mapping[str, Mapping[str, Any]]:
        return self._root_state_dtos

    def _score_frontier(self, item_meta, branch_results, depth=None):  # type: ignore[override]
        result = super()._score_frontier(item_meta, branch_results, depth)
        next_beam, newly_finished, _value_ms, _hit_depth, _hit_limit = result
        search_id = self._current_search_id()
        if search_id is None:
            return result

        for node in (*next_beam, *newly_finished):
            if (
                node.root_action_id is None
                or node.combat_depth != 1
                or is_continuation_decision(node.masked_emulator_dto)
            ):
                continue
            node_id = f"{search_id}:{node.branch_id}"
            self._root_state_dtos.setdefault(
                node_id,
                copy.deepcopy(dict(node.masked_emulator_dto)),
            )
        return result


class RootValueLoggingOracleCollector(BudgetedOracleCollector):
    """Budgeted Oracle collector that additionally emits root-action value samples."""

    async def collect(
        self,
        instance_id: str,
        root_decision: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> RootValueOracleCollectionResult:
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
        provenance = _oracle_provenance(self._policy, self._value_fn)  # noqa: SLF001
        engine = _RootValueCapturingOracleEngine(
            self._client,  # noqa: SLF001
            policy=self._policy,  # noqa: SLF001
            value_fn=self._value_fn,  # noqa: SLF001
            config=beam_config,
            stable_pruner=self._stable_pruner,  # noqa: SLF001
            trace_collector=trace_collector,
            exhaustive_root_actions=self.config.exhaustive_root_actions,
        )
        engine._allocator = self._branch_allocator  # noqa: SLF001
        search_result = await engine.search(
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
        samples = build_root_action_value_samples(
            trace_collector.events,
            targets,
            root_state_dtos=engine.root_state_dtos,
        )
        return RootValueOracleCollectionResult(
            search_result=search_result,
            trace=tuple(trace_collector.events),
            targets=targets,
            provenance=provenance,
            root_value_samples=samples,
        )


def build_root_action_value_samples(
    events: Sequence[SearchTraceEvent],
    targets: OracleTargets,
    *,
    root_state_dtos: Mapping[str, Mapping[str, Any]],
) -> tuple[RootActionValueSample, ...]:
    """Join root-action raw DTOs to the matching deeper Oracle RNG targets.

    At most one sample is emitted per root-action RNG outcome. When continuations produce
    more than one combat-depth-1 stable state, the state on the path to ``best_node_id``
    is selected. A censored/no-target outcome is retained for auditability with a null
    target. A non-terminal ``value_bootstrap`` whose best node is the root state itself is
    also converted to ``no_target`` because it contains no deeper-search supervision.
    Terminal post-states normalize their absent next-decision ID to ``None``.
    """

    resolved = [event for event in events if isinstance(event, ResolvedNodeTrace)]
    resolved_by_id = {node.node_id: node for node in resolved}
    candidates_by_key: dict[tuple[str, int], list[ResolvedNodeTrace]] = defaultdict(list)
    for node in resolved:
        if (
            node.root_action_id is not None
            and node.combat_depth == 1
            and node.value_is_fresh
            and node.node_id in root_state_dtos
        ):
            candidates_by_key[(node.root_action_id, node.rng_id)].append(node)

    samples: list[RootActionValueSample] = []
    for action_target in targets.root_actions:
        for outcome in action_target.rng_outcomes:
            candidates = candidates_by_key.get((action_target.action_id, outcome.rng_id), [])
            selected = _select_root_state_node(
                outcome.best_node_id,
                candidates=candidates,
                resolved_by_id=resolved_by_id,
            )
            if selected is None:
                continue
            dto = root_state_dtos.get(selected.node_id)
            if dto is None:
                continue

            target_value = outcome.value
            target_source = outcome.target_source
            terminal_reached = outcome.terminal_reached
            censored = outcome.censored
            censor_reason = outcome.censor_reason
            best_node_id = outcome.best_node_id
            if (
                target_source == "value_bootstrap"
                and best_node_id == selected.node_id
                and not selected.terminal
            ):
                target_value = None
                target_source = "no_target"
                terminal_reached = False
                censored = True
                censor_reason = "root_state_self_bootstrap_not_deeper_oracle_target"
                best_node_id = None

            samples.append(
                RootActionValueSample(
                    action_id=action_target.action_id,
                    rng_id=outcome.rng_id,
                    root_state_node_id=selected.node_id,
                    decision_point_id=selected.decision_point_id or None,
                    masked_emulator_dto=copy.deepcopy(dict(dto)),
                    target_value=target_value,
                    target_source=target_source,
                    terminal_reached=terminal_reached,
                    censored=censored,
                    censor_reason=censor_reason,
                    best_node_id=best_node_id,
                )
            )
    return tuple(samples)


def _select_root_state_node(
    best_node_id: str | None,
    *,
    candidates: Sequence[ResolvedNodeTrace],
    resolved_by_id: Mapping[str, ResolvedNodeTrace],
) -> ResolvedNodeTrace | None:
    if not candidates:
        return None
    candidate_by_id = {node.node_id: node for node in candidates}
    if best_node_id is not None:
        current = resolved_by_id.get(best_node_id)
        seen: set[str] = set()
        while current is not None and current.node_id not in seen:
            seen.add(current.node_id)
            selected = candidate_by_id.get(current.node_id)
            if selected is not None:
                return selected
            current = resolved_by_id.get(current.parent_node_id)

    return min(candidates, key=lambda node: (node.depth, node.node_id))
