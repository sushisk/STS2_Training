from __future__ import annotations

import unittest

from sts2_training.decision.beam_search import BeamNode, BeamSearchConfig, BeamSearchEngine
from sts2_training.decision.candidate_coverage import CoverageConstrainedPolicy
from sts2_training.decision.oracle_search import (
    BudgetedOracleCollector,
    OracleCollectionConfig,
    _OracleBeamSearchEngine,
    build_oracle_targets,
)
from sts2_training.decision.policy import ActionCandidate, PolicyModel
from sts2_training.decision.search_trace import (
    InMemorySearchTraceCollector,
    PolicyCandidateTrace,
    PolicyProposalTrace,
    ResolvedNodeTrace,
    SearchTraceEnd,
    SearchTraceStart,
    StablePruneNodeTrace,
    StablePruneTrace,
)
from sts2_training.decision.value import ValueModel


class _AllRequestedPolicy(PolicyModel):
    def propose(self, legal_actions, masked_emulator_dto, *, top_k):
        available = [action for action in legal_actions if action.get("is_available") is not False]
        return [
            ActionCandidate(action_id=action["action_id"])
            for action in available[:top_k]
        ]


class _StatefulPolicy(PolicyModel):
    def __init__(self) -> None:
        self.calls = 0

    def propose(self, legal_actions, masked_emulator_dto, *, top_k):
        self.calls += 1
        return [
            ActionCandidate(action_id=action["action_id"])
            for action in legal_actions[:top_k]
        ]


class _DummyValue(ValueModel):
    def evaluate_batch(self, masked_emulator_dtos):
        return [0.0 for _dto in masked_emulator_dtos]


def _resolved(
    *,
    search_id: str,
    node_id: str,
    parent_node_id: str,
    root_action_id: str,
    rng_id: int,
    value: float,
    terminal: bool = False,
    resolution: str = "expandable_stable",
    combat_depth: int = 1,
) -> ResolvedNodeTrace:
    branch_id = node_id.split(":", 1)[1]
    parent_branch_id = parent_node_id.split(":", 1)[1]
    return ResolvedNodeTrace(
        search_id=search_id,
        node_id=node_id,
        parent_node_id=parent_node_id,
        branch_id=branch_id,
        parent_branch_id=parent_branch_id,
        root_action_id=root_action_id,
        rng_id=rng_id,
        decision_point_id=f"d-{branch_id}",
        depth=combat_depth,
        combat_depth=combat_depth,
        continuation_steps=0,
        value=value,
        value_is_fresh=True,
        value_source="terminal" if terminal else "value_bootstrap",
        state_kind="terminal" if terminal else "stable",
        resolution="terminal" if terminal else resolution,
        terminal=terminal,
        action_id=f"act-{branch_id}",
        action_type="card",
        action={"action_id": f"act-{branch_id}", "action_type": "card"},
        policy_rank=0,
        policy_score=None,
        post_coverage_rank=0,
        candidate_source="policy",
    )


def _prune_node(
    *,
    node_id: str,
    value: float,
    root_action_id: str,
    rng_id: int,
    index: int,
    kept: bool,
) -> StablePruneNodeTrace:
    branch_id = node_id.split(":", 1)[1]
    return StablePruneNodeTrace(
        node_id=node_id,
        parent_node_id="search:root",
        branch_id=branch_id,
        parent_branch_id="root",
        frontier_index_before_prune=index,
        kept=kept,
        value=value,
        root_action_id=root_action_id,
        rng_id=rng_id,
        decision_point_id=f"d-{branch_id}",
        depth=1,
        combat_depth=1,
        continuation_steps=0,
        terminal=False,
        action_id=f"act-{branch_id}",
        action_type="card",
        action={"action_id": f"act-{branch_id}", "action_type": "card"},
        policy_rank=index,
        policy_score=None,
        post_coverage_rank=index,
        candidate_source="policy",
    )


class OracleCollectionConfigTest(unittest.TestCase):
    def test_rejects_common_rng_until_api_semantics_are_verified(self) -> None:
        with self.assertRaisesRegex(ValueError, "common-RNG"):
            OracleCollectionConfig(rng_sampling="common")

    def test_requires_oracle_beam_at_least_target_beam(self) -> None:
        with self.assertRaises(ValueError):
            OracleCollectionConfig(
                beam_config=BeamSearchConfig(beam_width=2),
                target_beam_width=3,
            )

    def test_from_beam_engine_copies_stateful_policy_and_records_inner_provenance(self) -> None:
        inner = _StatefulPolicy()
        runtime_policy = CoverageConstrainedPolicy(inner)
        engine = BeamSearchEngine(
            object(),
            policy=runtime_policy,
            value_fn=_DummyValue(),
        )

        oracle = BudgetedOracleCollector.from_beam_engine(engine)
        oracle._policy.propose(  # noqa: SLF001 - explicit state-isolation regression test
            [{"action_id": "a", "action_type": "card", "is_available": True}],
            {},
            top_k=1,
        )

        self.assertEqual(inner.calls, 0)
        self.assertIsNot(oracle._policy, runtime_policy)  # noqa: SLF001
        self.assertTrue(
            oracle._provenance.teacher_policy_class.endswith("CoverageConstrainedPolicy")  # noqa: SLF001
        )
        self.assertTrue(
            oracle._provenance.teacher_inner_policy_class.endswith("_StatefulPolicy")  # noqa: SLF001
        )
        self.assertTrue(
            oracle._provenance.teacher_coverage_policy_class.endswith(  # type: ignore[union-attr]  # noqa: SLF001
                "CoverageConstrainedPolicy"
            )
        )


class ExhaustiveRootProposalTest(unittest.TestCase):
    def test_exhaustive_root_ignores_descendant_top_k_limit(self) -> None:
        collector = InMemorySearchTraceCollector()
        engine = _OracleBeamSearchEngine(
            object(),
            policy=_AllRequestedPolicy(),
            value_fn=_DummyValue(),
            config=BeamSearchConfig(beam_width=4, top_k_actions=1, max_depth=2),
            trace_collector=collector,
            exhaustive_root_actions=True,
        )
        legal_actions = [
            {"action_id": "a", "action_type": "card", "is_available": True},
            {"action_id": "b", "action_type": "card", "is_available": True},
            {"action_id": "c", "action_type": "system", "is_available": True},
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

        items, metadata, _ms = engine._propose_frontier(
            [parent],
            search_id="search",
            proposal_step_index=0,
        )

        self.assertEqual([item["action_id"] for item in items], ["a", "b", "c"])
        self.assertEqual(len(metadata), 3)
        self.assertEqual(engine.config.top_k_actions, 1)

    def test_generic_value_model_terminal_prediction_is_not_promoted_to_terminal_target(self) -> None:
        collector = InMemorySearchTraceCollector()
        collector.record(
            SearchTraceStart(
                search_id="search",
                instance_id="inst",
                root_decision_point_id="d-root",
                beam_width=2,
                top_k_actions=1,
                max_depth=2,
                max_continuation_steps=8,
                time_budget_ms=None,
                pruner_name="value_top_k",
                pruner_version="1",
            )
        )
        engine = _OracleBeamSearchEngine(
            object(),
            policy=_AllRequestedPolicy(),
            value_fn=_DummyValue(),
            config=BeamSearchConfig(beam_width=2, top_k_actions=1, max_depth=2),
            trace_collector=collector,
            exhaustive_root_actions=False,
        )
        action = {"action_id": "a", "action_type": "card", "is_available": True}
        parent = BeamNode(
            branch_id="root",
            parent_branch_id="root",
            rng_id=0,
            decision_point_id="d-root",
            masked_emulator_dto={"legal_actions": [action]},
            depth=0,
            value=0.0,
            root_action_id=None,
        )

        engine._score_frontier(
            [(parent, ActionCandidate("a"), "branch", 1)],
            {
                "branch": {
                    "status": "completed",
                    "decision_point_id": "d-terminal",
                    "masked_emulator_dto": {
                        "terminal": True,
                        "outcome": "victory",
                        "legal_actions": [],
                    },
                }
            },
        )

        resolved = [event for event in collector.events if isinstance(event, ResolvedNodeTrace)]
        self.assertEqual(len(resolved), 1)
        self.assertTrue(resolved[0].terminal)
        self.assertEqual(resolved[0].value_source, "value_bootstrap")


class OracleTargetBuilderTest(unittest.TestCase):
    def test_wide_oracle_labels_node_that_runtime_value_top_k_would_drop(self) -> None:
        start = SearchTraceStart(
            search_id="search",
            instance_id="inst",
            root_decision_point_id="d-root",
            beam_width=2,
            top_k_actions=1,
            max_depth=2,
            max_continuation_steps=8,
            time_budget_ms=None,
            pruner_name="value_top_k",
            pruner_version="1",
        )
        legal = (
            {"action_id": "A", "action_type": "card", "is_available": True},
            {"action_id": "B", "action_type": "card", "is_available": True},
            {"action_id": "C", "action_type": "system", "is_available": True},
        )
        proposal = PolicyProposalTrace(
            search_id="search",
            proposal_step_id="search:proposal:0:0",
            parent_node_id="search:root",
            parent_branch_id="root",
            decision_point_id="d-root",
            legal_actions=legal,
            candidates=(
                PolicyCandidateTrace("A", legal[0], "a", 1, 0, None, 0, "policy"),
                PolicyCandidateTrace("B", legal[1], "b", 2, 1, None, 1, "policy"),
            ),
        )
        resolved_a = _resolved(
            search_id="search",
            node_id="search:a",
            parent_node_id="search:root",
            root_action_id="A",
            rng_id=1,
            value=5.0,
        )
        resolved_b = _resolved(
            search_id="search",
            node_id="search:b",
            parent_node_id="search:root",
            root_action_id="B",
            rng_id=2,
            value=6.0,
        )
        terminal_a = _resolved(
            search_id="search",
            node_id="search:a1",
            parent_node_id="search:a",
            root_action_id="A",
            rng_id=1,
            value=10.0,
            terminal=True,
            combat_depth=2,
        )
        prune = StablePruneTrace(
            search_id="search",
            prune_step_id="search:prune:0",
            phase="stable_frontier",
            k=2,
            frontier_size=2,
            pruner_name="value_top_k",
            pruner_version="1",
            max_depth=2,
            depths_completed=0,
            remaining_time_ms=None,
            nodes=(
                _prune_node(
                    node_id="search:a",
                    value=5.0,
                    root_action_id="A",
                    rng_id=1,
                    index=0,
                    kept=True,
                ),
                _prune_node(
                    node_id="search:b",
                    value=6.0,
                    root_action_id="B",
                    rng_id=2,
                    index=1,
                    kept=True,
                ),
            ),
        )
        end = SearchTraceEnd(
            search_id="search",
            reason="max_depth",
            best_root_action_id="A",
            best_value=10.0,
            depths_completed=2,
            nodes_expanded=3,
            branches_created=3,
        )

        targets = build_oracle_targets(
            [start, proposal, resolved_a, resolved_b, prune, terminal_a, end],
            target_beam_width=1,
            exhaustive_root_actions=False,
        )

        stable = {target.node_id: target for target in targets.stable_nodes}
        self.assertFalse(stable["search:a"].baseline_would_keep)
        self.assertTrue(stable["search:a"].oracle_kept)
        self.assertEqual(stable["search:a"].target_value, 10.0)
        self.assertEqual(stable["search:a"].target_source, "terminal")
        self.assertFalse(stable["search:a"].censored)
        self.assertTrue(stable["search:b"].baseline_would_keep)
        self.assertIsNone(stable["search:b"].target_value)
        self.assertEqual(stable["search:b"].target_source, "no_target")

        root = {target.action_id: target for target in targets.root_actions}
        self.assertEqual(root["A"].estimated_q, 10.0)
        self.assertEqual(root["A"].target_source, "terminal")
        self.assertEqual(root["B"].estimated_q, 6.0)
        self.assertTrue(root["B"].censored)
        self.assertFalse(root["C"].evaluated)
        self.assertEqual(root["C"].censor_reason, "policy_candidate_limit")

    def test_oracle_pruned_node_does_not_use_its_current_value_as_target(self) -> None:
        start = SearchTraceStart(
            search_id="search",
            instance_id="inst",
            root_decision_point_id="d-root",
            beam_width=1,
            top_k_actions=1,
            max_depth=2,
            max_continuation_steps=8,
            time_budget_ms=None,
            pruner_name="value_top_k",
            pruner_version="1",
        )
        legal = ({"action_id": "A", "action_type": "card", "is_available": True},)
        proposal = PolicyProposalTrace(
            search_id="search",
            proposal_step_id="p",
            parent_node_id="search:root",
            parent_branch_id="root",
            decision_point_id="d-root",
            legal_actions=legal,
            candidates=(PolicyCandidateTrace("A", legal[0], "a", 1, 0, None, 0, "policy"),),
        )
        resolved = _resolved(
            search_id="search",
            node_id="search:a",
            parent_node_id="search:root",
            root_action_id="A",
            rng_id=1,
            value=99.0,
        )
        prune = StablePruneTrace(
            search_id="search",
            prune_step_id="prune",
            phase="stable_frontier",
            k=1,
            frontier_size=1,
            pruner_name="test",
            pruner_version="1",
            max_depth=2,
            depths_completed=0,
            remaining_time_ms=None,
            nodes=(
                _prune_node(
                    node_id="search:a",
                    value=99.0,
                    root_action_id="A",
                    rng_id=1,
                    index=0,
                    kept=False,
                ),
            ),
        )
        end = SearchTraceEnd("search", "beam_exhausted", None, None, 1, 1, 1)

        targets = build_oracle_targets(
            [start, proposal, resolved, prune, end],
            target_beam_width=1,
            exhaustive_root_actions=True,
        )

        target = targets.stable_nodes[0]
        self.assertIsNone(target.target_value)
        self.assertEqual(target.target_source, "no_target")
        self.assertEqual(target.censor_reason, "oracle_pruned_before_followup")


if __name__ == "__main__":
    unittest.main()