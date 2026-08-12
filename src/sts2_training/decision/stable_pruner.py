"""Stable/resolved Combat frontier pruning seam.

This module deliberately does not own continuation handling, policy candidate limits,
or Whole Run active-branch capacity. ``BeamSearchEngine`` remains responsible for those
lifecycles and passes an already ordered stable frontier to ``StableFrontierPruner``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar


class _ValuedNode(Protocol):
    value: float


NodeT = TypeVar("NodeT", bound=_ValuedNode)


@dataclass(frozen=True)
class StablePruneContext:
    """Immutable search context available to current and future stable pruners."""

    search_id: str
    prune_step_id: str
    phase: str
    beam_width: int
    max_depth: int
    depths_completed: int
    remaining_time_ms: float | None


class StableFrontierPruner:
    """Select at most ``k`` nodes from one ordered stable/resolved frontier."""

    name = "stable_frontier_pruner"
    version = "1"

    def select(
        self,
        frontier: Sequence[NodeT],
        *,
        k: int,
        context: StablePruneContext,
    ) -> list[NodeT]:
        raise NotImplementedError


class ValueTopKPruner(StableFrontierPruner):
    """Exact baseline: stable-sort by descending ``node.value`` and keep top-K."""

    name = "value_top_k"
    version = "1"

    def select(
        self,
        frontier: Sequence[NodeT],
        *,
        k: int,
        context: StablePruneContext,
    ) -> list[NodeT]:
        del context
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        # Python's sort is stable, preserving the existing tie-order behavior.
        ranked = sorted(frontier, key=lambda node: node.value, reverse=True)
        return ranked[:k]
