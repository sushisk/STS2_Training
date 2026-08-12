from __future__ import annotations

import time
import unittest
from dataclasses import FrozenInstanceError

import sts2_training.decision as decision
from sts2_training.decision.beam_search import BeamNode, BeamSearchConfig, BeamSearchEngine
from sts2_training.decision.policy import PolicyModel
from sts2_training.decision.search_trace import InMemorySearchTraceCollector, StablePruneTrace
from sts2_training.decision.stable_pruner import (
    STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
    StableFrontierPruner,
    StablePruneContext,
    StablePruneNodeView,
    ValueTopKPruner,
)
from sts2_training.decision.value import ValueModel


class _NoopPolicy(PolicyModel):
    def propose(self, legal_actions, masked_emulator_dto, *, top_k):
        return []


class _ZeroValue(ValueModel):
    def evaluate_batch(self, masked_emulator_dtos):
        return [0.0 for _dto in masked_emulator_dtos]


class _FixedIndexPruner(StableFrontierPruner):
    name = "fixed_indices"
    version = "test"

    def __init__(self, selected: object) -> None:
        self.selected = selected
        self.frontier: tuple[StablePruneNodeView, ...] | None = None
        self.context: StablePruneContext | None = None
        self.calls = 0

    def select(self, frontier, *, k, context):
        del k
        self.calls += 1
        self.frontier = tuple(frontier)
        self.context = context
        return self.selected


def _stable_node(
    branch_id: str,
    value: float,
    *,
    root_action_id: str = "root-action",
    depth: int = 2,
    combat_depth: int = 1,
    continuation_steps: int = 0,
    terminal: bool = False,
    action_type: str = "card",
    policy_rank: int | None = 1,
    policy_score: float | None = 0.25,
    post_coverage_rank: int | None = 0,
    candidate_source: str | None = "policy",
) -> BeamNode:
    return BeamNode(
        branch_id=branch_id,
        parent_branch_id="root",
        rng_id=11,
        decision_point_id=f"decision-{branch_id}",
        masked_emulator_dto={
            "legal_actions": [
                {"action_id": "next", "action_type": "card", "is_available": True}
            ]
        },
        depth=depth,
        value=value,
        root_action_id=root_action_id,
        combat_depth=combat_depth,
        continuation_steps=continuation_steps,
        branch_log=("private",),
        terminal=terminal,
        action_id=f"action-{branch_id}",
        action_type=action_type,
        action={"action_id": f"action-{branch_id}", "secret": "payload"},
        policy_rank=policy_rank,
        policy_score=policy_score,
        post_coverage_rank=post_coverage_rank,
        candidate_source=candidate_source,
    )


def _continuation_node() -> BeamNode:
    return BeamNode(
        branch_id="continuation",
        parent_branch_id="root",
        rng_id=12,
        decision_point_id="decision-continuation",
        masked_emulator_dto={
            "legal_actions": [
                {
                    "action_id": "choose-target",
                    "action_type": "choice_target",
                    "is_available": True,
                }
            ]
        },
        depth=2,
        value=123.0,
        root_action_id="root-action",
        combat_depth=1,
        continuation_steps=1,
        action_id="choose-target",
        action_type="choice_target",
        policy_rank=0,
        post_coverage_rank=0,
        candidate_source="policy",
    )


def _engine(
    pruner: StableFrontierPruner,
    *,
    collector: InMemorySearchTraceCollector | None = None,
) -> BeamSearchEngine:
    return BeamSearchEngine(
        object(),
        policy=_NoopPolicy(),
        value_fn=_ZeroValue(),
        config=BeamSearchConfig(beam_width=2, top_k_actions=1, max_depth=4),
        stable_pruner=pruner,
        trace_collector=collector,
    )


def _prune(
    engine: BeamSearchEngine,
    frontier: list[BeamNode],
    *,
    phase: str = "stable_frontier",
):
    return engine._prune_stable_frontier(  # noqa: SLF001 - public seam integration contract
        frontier,
        search_id="search",
        prune_step_index=3,
        phase=phase,
        search_deadline=time.monotonic() + 5.0,
        depths_completed=2,
    )


class StablePrunePublicContractTest(unittest.TestCase):
    def test_public_schema_and_exports_are_fixed_at_v1(self) -> None:
        self.assertEqual(STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION, 1)
        self.assertEqual(decision.STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION, 1)
        self.assertIs(decision.StablePruneNodeView, StablePruneNodeView)
        self.assertIs(decision.StablePruneContext, StablePruneContext)
        self.assertIs(decision.StableFrontierPruner, StableFrontierPruner)

    def test_view_is_immutable(self) -> None:
        view = StablePruneNodeView(
            value=1.0,
            root_action_id="root-action",
            depth=1,
            combat_depth=1,
            continuation_steps=0,
            terminal=False,
            action_type="card",
            policy_rank=0,
            policy_score=0.5,
            post_coverage_rank=0,
            candidate_source="policy",
        )
        with self.assertRaises(FrozenInstanceError):
            view.value = 2.0  # type: ignore[misc]

    def test_custom_pruner_receives_only_views_and_return_order_maps_to_beam_nodes(self) -> None:
        pruner = _FixedIndexPruner([2, 0])
        frontier = [
            _stable_node("a", 1.0),
            _stable_node("b", 2.0),
            _stable_node("c", 3.0),
        ]

        selected = _prune(_engine(pruner), frontier)

        self.assertEqual([node.branch_id for node in selected], ["c", "a"])
        assert pruner.frontier is not None
        self.assertEqual([view.value for view in pruner.frontier], [1.0, 2.0, 3.0])
        for view in pruner.frontier:
            self.assertIsInstance(view, StablePruneNodeView)
            for private_name in (
                "branch_id",
                "parent_branch_id",
                "decision_point_id",
                "rng_id",
                "masked_emulator_dto",
                "action",
                "action_id",
                "branch_log",
            ):
                self.assertFalse(hasattr(view, private_name), private_name)
        assert pruner.context is not None
        self.assertEqual(pruner.context.beam_width, 2)
        self.assertEqual(pruner.context.phase, "stable_frontier")

    def test_invalid_returned_indices_fail_fast(self) -> None:
        frontier = [
            _stable_node("a", 1.0),
            _stable_node("b", 2.0),
            _stable_node("c", 3.0),
        ]
        invalid = {
            "duplicate": [0, 0],
            "negative": [-1],
            "out_of_range": [3],
            "bool": [True],
            "non_int": [0.0],
            "more_than_k": [0, 1, 2],
            "not_a_list": (0,),
        }
        for label, selected in invalid.items():
            with self.subTest(label=label):
                with self.assertRaises(RuntimeError):
                    _prune(_engine(_FixedIndexPruner(selected)), frontier)

    def test_runtime_views_equal_trace_replay_views_field_by_field(self) -> None:
        collector = InMemorySearchTraceCollector()
        pruner = _FixedIndexPruner([1, 0])
        frontier = [
            _stable_node(
                "a",
                4.5,
                root_action_id="A",
                depth=3,
                combat_depth=2,
                continuation_steps=0,
                action_type="potion",
                policy_rank=None,
                policy_score=None,
                post_coverage_rank=2,
                candidate_source="structural_coverage",
            ),
            _stable_node(
                "b",
                7.0,
                root_action_id="B",
                depth=4,
                combat_depth=3,
                continuation_steps=1,
                terminal=True,
                action_type="card",
                policy_rank=0,
                policy_score=0.75,
                post_coverage_rank=0,
                candidate_source="policy",
            ),
        ]

        selected = _prune(_engine(pruner, collector=collector), frontier)

        self.assertEqual([node.branch_id for node in selected], ["b", "a"])
        self.assertEqual(len(collector.events), 1)
        trace = collector.events[0]
        self.assertIsInstance(trace, StablePruneTrace)
        assert isinstance(trace, StablePruneTrace)
        assert pruner.frontier is not None
        self.assertEqual(trace.node_views(), pruner.frontier)
        self.assertEqual(
            [node.frontier_index_before_prune for node in trace.nodes],
            list(range(len(frontier))),
        )
        self.assertEqual([node.kept for node in trace.nodes], [True, True])
        runtime_context = pruner.context
        assert runtime_context is not None
        replay_context = trace.to_prune_context()
        self.assertEqual(replay_context, runtime_context)
        narrowed = trace.to_prune_context(beam_width=1)
        self.assertEqual(narrowed.beam_width, 1)
        self.assertEqual(narrowed.search_id, runtime_context.search_id)
        self.assertEqual(narrowed.prune_step_id, runtime_context.prune_step_id)
        self.assertEqual(narrowed.phase, runtime_context.phase)
        self.assertEqual(narrowed.max_depth, runtime_context.max_depth)
        self.assertEqual(narrowed.depths_completed, runtime_context.depths_completed)
        self.assertEqual(narrowed.remaining_time_ms, runtime_context.remaining_time_ms)

    def test_continuation_inherited_value_is_rejected_before_pruner_call(self) -> None:
        pruner = _FixedIndexPruner([0])
        with self.assertRaisesRegex(RuntimeError, "continuation"):
            _prune(_engine(pruner), [_continuation_node()])
        self.assertEqual(pruner.calls, 0)

    def test_value_top_k_returns_stable_ordered_indices(self) -> None:
        views = (
            _stable_node("a", 5.0),
            _stable_node("b", 5.0),
            _stable_node("c", 8.0),
        )
        public_views = tuple(
            StablePruneNodeView(
                value=node.value,
                root_action_id=node.root_action_id,
                depth=node.depth,
                combat_depth=node.combat_depth,
                continuation_steps=node.continuation_steps,
                terminal=node.terminal,
                action_type=node.action_type,
                policy_rank=node.policy_rank,
                policy_score=node.policy_score,
                post_coverage_rank=node.post_coverage_rank,
                candidate_source=node.candidate_source,
            )
            for node in views
        )
        context = StablePruneContext(
            search_id="search",
            prune_step_id="prune",
            phase="stable_frontier",
            beam_width=3,
            max_depth=4,
            depths_completed=1,
            remaining_time_ms=None,
        )

        self.assertEqual(
            ValueTopKPruner().select(public_views, k=3, context=context),
            [2, 0, 1],
        )


if __name__ == "__main__":
    unittest.main()
