"""Stable/resolved Combat frontier pruning public contract.

This module deliberately does not own continuation handling, policy candidate limits,
or Whole Run active-branch capacity. ``BeamSearchEngine`` remains responsible for those
lifecycles and passes an already ordered stable frontier to ``StableFrontierPruner``.

The public seam exposes only ``StablePruneNodeView``. Internal ``BeamNode`` identity,
DTO/action payloads, branch/node IDs, RNG identity, logs, and capacity state stay outside
the pruning API.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION = 1
_CANDIDATE_SOURCES = frozenset({"policy", "structural_coverage"})


@dataclass(frozen=True)
class StablePruneNodeView:
    """Immutable public input for one stable-frontier pruning candidate.

    Contract v1 field semantics:

    - ``value`` is the finite current score for a stable/resolved/terminal node. A
      continuation node's inherited/stale value must never enter this seam.
    - ``root_action_id`` is an opaque grouping key scoped to one search only; it is not a
      global learned identity.
    - ``depth`` is Beam transition depth; ``combat_depth`` counts non-continuation Combat
      actions; ``continuation_steps`` is the current continuation-safety counter.
    - ``terminal`` identifies terminal stable/resolved nodes.
    - ``action_type`` is only the coarse semantic type of the action that produced this
      node. Full action payloads are deliberately excluded.
    - ``policy_rank`` is the inner-policy 0-based rank, or ``None`` when structural
      coverage inserted the candidate without an inner-policy rank.
    - ``policy_score`` is a finite numeric score when available, otherwise ``None``.
    - ``post_coverage_rank`` is the 0-based rank after structural coverage, or ``None``
      only when provenance is unavailable for a synthetic/legacy node.
    - ``candidate_source`` is ``"policy"`` or ``"structural_coverage"``; ``None`` is
      reserved for synthetic/legacy nodes whose provenance was not recorded.

    Changing this field set or any of these semantics requires incrementing
    ``STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION``.
    """

    value: float
    root_action_id: str | None
    depth: int
    combat_depth: int
    continuation_steps: int
    terminal: bool
    action_type: str | None
    policy_rank: int | None
    policy_score: float | None
    post_coverage_rank: int | None
    candidate_source: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _finite_float(self.value, "value"))
        _optional_non_empty_str(self.root_action_id, "root_action_id")
        _non_negative_int(self.depth, "depth")
        _non_negative_int(self.combat_depth, "combat_depth")
        _non_negative_int(self.continuation_steps, "continuation_steps")
        if not isinstance(self.terminal, bool):
            raise TypeError("terminal must be a bool")
        _optional_non_empty_str(self.action_type, "action_type")
        _optional_non_negative_int(self.policy_rank, "policy_rank")
        if self.policy_score is not None:
            object.__setattr__(
                self,
                "policy_score",
                _finite_float(self.policy_score, "policy_score"),
            )
        _optional_non_negative_int(self.post_coverage_rank, "post_coverage_rank")
        if self.candidate_source is not None:
            if not isinstance(self.candidate_source, str):
                raise TypeError("candidate_source must be a string or None")
            if self.candidate_source not in _CANDIDATE_SOURCES:
                raise ValueError(
                    "candidate_source must be 'policy', 'structural_coverage', or None"
                )


@dataclass(frozen=True)
class StablePruneContext:
    """Immutable context for exactly one stable-pruning invocation.

    ``beam_width`` is that invocation's target K/runtime beam width. ``max_depth`` and
    ``depths_completed`` retain Beam Search's current budget semantics.
    ``remaining_time_ms`` is either ``None`` or a finite non-negative number.
    """

    search_id: str
    prune_step_id: str
    phase: str
    beam_width: int
    max_depth: int
    depths_completed: int
    remaining_time_ms: float | None

    def __post_init__(self) -> None:
        _non_empty_str(self.search_id, "search_id")
        _non_empty_str(self.prune_step_id, "prune_step_id")
        _non_empty_str(self.phase, "phase")
        _positive_int(self.beam_width, "beam_width")
        _positive_int(self.max_depth, "max_depth")
        _non_negative_int(self.depths_completed, "depths_completed")
        if self.remaining_time_ms is not None:
            remaining = _finite_float(self.remaining_time_ms, "remaining_time_ms")
            if remaining < 0:
                raise ValueError("remaining_time_ms must be non-negative when provided")
            object.__setattr__(self, "remaining_time_ms", remaining)


class StableFrontierPruner:
    """Select ordered survivor indices from one ordered stable/resolved frontier.

    The input sequence order is authoritative. Returned indices must be unique integers in
    ``[0, len(frontier))`` and their order *is* survivor order. Runtime validation and the
    index-to-internal-node mapping are owned by ``BeamSearchEngine``.
    """

    name = "stable_frontier_pruner"
    version = "1"

    def select(
        self,
        frontier: Sequence[StablePruneNodeView],
        *,
        k: int,
        context: StablePruneContext,
    ) -> list[int]:
        raise NotImplementedError


class ValueTopKPruner(StableFrontierPruner):
    """Exact baseline: stable descending-value sort over frontier indices."""

    name = "value_top_k"
    version = "1"

    def select(
        self,
        frontier: Sequence[StablePruneNodeView],
        *,
        k: int,
        context: StablePruneContext,
    ) -> list[int]:
        del context
        _positive_int(k, "k")
        # Python's sort is stable, so equal values preserve authoritative frontier order.
        ranked_indices = sorted(
            range(len(frontier)),
            key=lambda index: frontier[index].value,
            reverse=True,
        )
        return ranked_indices[:k]


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return normalized


def _non_empty_str(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _optional_non_empty_str(value: object, name: str) -> None:
    if value is not None:
        _non_empty_str(value, name)


def _positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _non_negative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _optional_non_negative_int(value: object, name: str) -> None:
    if value is not None:
        _non_negative_int(value, name)
