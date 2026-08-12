from __future__ import annotations

import time
import unittest

from sts2_training.decision.beam_search import BeamNode, BeamSearchConfig, BeamSearchEngine
from sts2_training.decision.candidate_coverage import CoverageConstrainedPolicy
from sts2_training.decision.policy import ActionCandidate, PolicyModel
from sts2_training.decision.search_trace import (
    InMemorySearchTraceCollector,
    PolicyProposalTrace,
    SearchTraceEnd,
    SearchTraceStart,
    StablePruneTrace,
)
from sts2_training.decision.stable_pruner import (
    StableFrontierPruner,
    StablePruneContext,
    StablePruneNodeView,
    ValueTopKPruner,
)
from sts2_training.decision.value import ValueModel


class _RankedPolicy(PolicyModel):
    def __init__(self, action_ids: list[str]) -> None:
        self._action_ids = action_ids

    def propose(self, legal_actions, masked_emulator_dto, *, top_k):
        return [ActionCandidate(action_id=action_id) for action_id in self._action_ids[:top_k]]


class _DummyValue(ValueModel):
    def evaluate_batch(self, masked_emulator_dtos):
        return [0.0 for _dto in masked_emulator_dtos]


class _ReversePruner(StableFrontierPruner):
    name = "reverse"
    version = "test"

    def __init__(self) -> None:
        self.seen: list[StablePruneNodeView] = []

    def select(self, frontier, *, k, context):
        del context
        self.seen = list(frontier)
        return list(reversed(range(len(frontier))))[:k]


class _TerminalClient:
    instance_type = "combat"

    async def emulate_actions(
        self,
        instance_id,
        items,
        *,
        timeout_s,
        simulation_options,
    ):
        return {
            "branch_results": {
                item["branch_id"]: {
                    "status": "completed",
                    "decision_point_id": "d-terminal",
                    "masked_emulator_dto": {
                        "terminal": True,
                        "outcome": "victory",
                        "legal_actions": [],
                    },
                }
                for item in items
            }
        }

    async def cancel_branches(self, instance_id, branch_ids, *, timeout_s):
        return None

    async def release_branches(self, instance_id, branch_ids, *, timeout_s):
        return None


def _node(branch_id: str, value: float, *, parent: str = "root") -> BeamNode:
    return BeamNode(
        branch_id=branch_id,
        parent_branch_id=parent,
        rng_id=1,
        decision_point_id=f"d-{branch_id}",
        masked_emulator_dto={"legal_actions": []},
        depth=1,
        value=value,
        root_action_id="root-action",
        action_id=f"action-{branch_id}",
        action_type="card",
        action={"action_id": f"action-{branch_id}", "action_type": "card"},
        policy_rank=0,
        post_coverage_rank=0,
        candidate_source="policy",
    )


def _view(value: float) -> StablePruneNodeView:
    return StablePruneNodeView(
        value=value,
        root_action_id="root-action",
        depth=1,
        combat_depth=1,
        continuation_steps=0,
        terminal=False,
        action_type="card",
        policy_rank=0,
        policy_score=None,
        post_coverage_rank=0,
        candidate_source="policy",
    )


class StableFrontierPrunerTest(unittest.TestCase):
    def _context(self) -> StablePruneContext:
        return StablePruneContext(
            search_id="search",
            prune_step_id="search:prune:0",
            phase="stable_frontier",
            beam_width=2,
            max_depth=2,
            depths_completed=0,
            remaining_time_ms=1000.0,
        )

    def test_value_top_k_preserves_existing_tie_order(self) -> None:
        frontier = [_view(5.0), _view(5.0), _view(1.0)]
        before = list(frontier)

        selected = ValueTopKPruner().select(frontier, k=2, context=self._context())

        self.assertEqual(selected, [0, 1])
        self.assertEqual(frontier, before)

    def test_rejects_non_positive_k(self) -> None:
        with self.assertRaises(ValueError):
            ValueTopKPruner().select([], k=0, context=self._context())


class CoverageProvenanceTest(unittest.TestCase):
    def test_distinguishes_inner_policy_from_structural_coverage(self) -> None:
        legal_actions = [
            {"action_id": "p1", "action_type": "potion", "is_available": True},
            {"action_id": "p2", "action_type": "potion", "is_available": True},
            {"action_id": "p3", "action_type": "potion", "is_available": True},
            {"action_id": "card", "action_type": "card", "is_available": True},
            {"action_id": "end", "action_type": "system", "is_available": True},
        ]
        policy = CoverageConstrainedPolicy(_RankedPolicy(["p1", "p2", "p3"]))

        proposals = policy.propose_with_provenance(legal_actions, {}, top_k=4)
        by_id = {proposal.action_id: proposal for proposal in proposals}

        self.assertEqual(by_id["p1"].candidate_source, "policy")
        self.assertEqual(by_id["p1"].policy_rank, 0)
        self.assertEqual(by_id["card"].candidate_source, "structural_coverage")
        self.assertIsNone(by_id["card"].policy_rank)
        self.assertEqual(by_id["end"].candidate_source, "structural_coverage")
        self.assertEqual(
            [proposal.post_coverage_rank for proposal in proposals],
            list(range(len(proposals))),
        )


class BeamSearchInstrumentationTest(unittest.TestCase):
    def _engine(
        self,
        *,
        policy: PolicyModel | None = None,
        pruner: StableFrontierPruner | None = None,
        collector: InMemorySearchTraceCollector | None = None,
    ) -> BeamSearchEngine:
        return BeamSearchEngine(
            object(),
            policy=policy or _RankedPolicy(["card", "end"]),
            value_fn=_DummyValue(),
            config=BeamSearchConfig(beam_width=2, top_k_actions=2, max_depth=2),
            stable_pruner=pruner,
            trace_collector=collector,
        )

    def test_pruner_receives_ordered_frontier_and_controls_only_selection(self) -> None:
        collector = InMemorySearchTraceCollector()
        pruner = _ReversePruner()
        engine = self._engine(pruner=pruner, collector=collector)
        frontier = [_node("a", 1.0), _node("b", 2.0), _node("c", 3.0)]

        selected = engine._prune_stable_frontier(
            frontier,
            search_id="search",
            prune_step_index=0,
            phase="stable_frontier",
            search_deadline=time.monotonic() + 1.0,
            depths_completed=1,
        )

        self.assertEqual([view.value for view in pruner.seen], [1.0, 2.0, 3.0])
        self.assertTrue(all(isinstance(view, StablePruneNodeView) for view in pruner.seen))
        self.assertEqual([node.branch_id for node in selected], ["c", "b"])
        self.assertEqual(len(collector.events), 1)
        event = collector.events[0]
        self.assertIsInstance(event, StablePruneTrace)
        assert isinstance(event, StablePruneTrace)
        self.assertEqual(event.search_id, "search")
        self.assertEqual(event.prune_step_id, "search:prune:0")
        self.assertEqual(event.k, 2)
        self.assertEqual(event.pruner_name, "reverse")
        self.assertEqual(
            [node.frontier_index_before_prune for node in event.nodes],
            [0, 1, 2],
        )
        self.assertEqual([node.kept for node in event.nodes], [False, True, True])
        self.assertEqual(event.nodes[0].node_id, "search:a")
        self.assertEqual(event.nodes[0].parent_node_id, "search:root")
        self.assertEqual(event.node_views(), tuple(pruner.seen))
        replay_context = event.to_prune_context()
        self.assertEqual(replay_context.beam_width, 2)
        self.assertEqual(replay_context.search_id, "search")

    def test_policy_trace_records_legal_actions_and_coverage_provenance(self) -> None:
        collector = InMemorySearchTraceCollector()
        policy = CoverageConstrainedPolicy(_RankedPolicy(["p1", "p2", "p3"]))
        engine = self._engine(policy=policy, collector=collector)
        legal_actions = [
            {"action_id": "p1", "action_type": "potion", "is_available": True},
            {"action_id": "p2", "action_type": "potion", "is_available": True},
            {"action_id": "p3", "action_type": "potion", "is_available": True},
            {"action_id": "card", "action_type": "card", "is_available": True},
            {"action_id": "end", "action_type": "system", "is_available": True},
        ]
        parent = BeamNode(
            branch_id="root",
            parent_branch_id="root",
            rng_id=0,
            decision_point_id="d-root",
            masked_emulator_dto={"legal_actions": legal_actions},
            depth=0,
            value=0.0,
            root_action_id=None,
        )

        items, metadata, _policy_ms = engine._propose_frontier(
            [parent],
            search_id="search",
            proposal_step_index=0,
        )

        self.assertEqual(len(items), 2)
        self.assertEqual(len(metadata), 2)
        self.assertEqual(len(collector.events), 1)
        event = collector.events[0]
        self.assertIsInstance(event, PolicyProposalTrace)
        assert isinstance(event, PolicyProposalTrace)
        self.assertEqual(event.search_id, "search")
        self.assertEqual(event.proposal_step_id, "search:proposal:0:0")
        self.assertEqual(len(event.legal_actions), 5)
        self.assertEqual(
            [candidate.post_coverage_rank for candidate in event.candidates],
            [0, 1],
        )
        self.assertEqual(
            [candidate.candidate_source for candidate in event.candidates],
            ["structural_coverage", "structural_coverage"],
        )


class BeamSearchCompletionTraceTest(unittest.IsolatedAsyncioTestCase):
    async def test_normal_beam_search_emits_exactly_one_completion_event(self) -> None:
        collector = InMemorySearchTraceCollector()
        engine = BeamSearchEngine(
            _TerminalClient(),
            policy=_RankedPolicy(["card"]),
            value_fn=_DummyValue(),
            config=BeamSearchConfig(beam_width=1, top_k_actions=1, max_depth=2),
            trace_collector=collector,
        )
        decision = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": {
                "legal_actions": [
                    {"action_id": "card", "action_type": "card", "is_available": True}
                ]
            },
        }

        result = await engine.search("inst", decision, timeout_s=5.0)

        starts = [event for event in collector.events if isinstance(event, SearchTraceStart)]
        ends = [event for event in collector.events if isinstance(event, SearchTraceEnd)]
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(ends), 1)
        self.assertIs(collector.events[-1], ends[0])
        self.assertEqual(ends[0].search_id, starts[0].search_id)
        self.assertEqual(ends[0].reason, result.reason)
        self.assertEqual(ends[0].best_root_action_id, "card")


if __name__ == "__main__":
    unittest.main()
