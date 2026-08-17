from __future__ import annotations

import unittest

from sts2_training.decision.oracle_search import build_oracle_targets
from sts2_training.decision.search_trace import (
    BranchFaultTrace,
    PolicyCandidateTrace,
    PolicyProposalTrace,
    ResolvedNodeTrace,
    SearchTraceEnd,
    SearchTraceStart,
    StablePruneNodeTrace,
    StablePruneTrace,
)


def _start() -> SearchTraceStart:
    return SearchTraceStart(
        search_id="search",
        instance_id="inst",
        root_decision_point_id="d-root",
        beam_width=2,
        top_k_actions=2,
        max_depth=3,
        max_continuation_steps=8,
        time_budget_ms=None,
        pruner_name="value_top_k",
        pruner_version="1",
    )


def _root_proposal() -> PolicyProposalTrace:
    action = {"action_id": "A", "action_type": "card", "is_available": True}
    return PolicyProposalTrace(
        search_id="search",
        proposal_step_id="search:proposal:0:0",
        parent_node_id="search:root",
        parent_branch_id="root",
        decision_point_id="d-root",
        legal_actions=(action,),
        candidates=(
            PolicyCandidateTrace(
                action_id="A",
                action=action,
                branch_id="a",
                rng_id=1,
                policy_rank=0,
                policy_score=None,
                post_coverage_rank=0,
                candidate_source="policy",
            ),
        ),
    )


def _resolved(
    *,
    node_id: str,
    parent_node_id: str,
    branch_id: str,
    parent_branch_id: str,
    rng_id: int,
    value: float,
    terminal: bool,
) -> ResolvedNodeTrace:
    return ResolvedNodeTrace(
        search_id="search",
        node_id=node_id,
        parent_node_id=parent_node_id,
        branch_id=branch_id,
        parent_branch_id=parent_branch_id,
        root_action_id="A",
        rng_id=rng_id,
        decision_point_id=f"d-{branch_id}",
        depth=1 if parent_branch_id == "root" else 2,
        combat_depth=1 if parent_branch_id == "root" else 2,
        continuation_steps=0,
        value=value,
        value_is_fresh=True,
        value_source="terminal" if terminal else "value_bootstrap",
        state_kind="terminal" if terminal else "stable",
        resolution="terminal" if terminal else "expandable_stable",
        terminal=terminal,
        action_id=f"act-{branch_id}",
        action_type="card",
        action={"action_id": f"act-{branch_id}", "action_type": "card"},
        policy_rank=0,
        policy_score=None,
        post_coverage_rank=0,
        candidate_source="policy",
    )


def _fault(
    *,
    branch_id: str,
    parent_node_id: str,
    parent_branch_id: str,
    rng_id: int,
) -> BranchFaultTrace:
    return BranchFaultTrace(
        search_id="search",
        node_id=f"search:{branch_id}",
        parent_node_id=parent_node_id,
        branch_id=branch_id,
        parent_branch_id=parent_branch_id,
        root_action_id="A",
        rng_id=rng_id,
        depth=1 if parent_branch_id == "root" else 2,
        combat_depth=1 if parent_branch_id == "root" else 2,
        continuation_steps=0,
        action_id=f"act-{branch_id}",
        action_type="card",
        action={"action_id": f"act-{branch_id}", "action_type": "card"},
        policy_rank=1,
        policy_score=None,
        post_coverage_rank=1,
        candidate_source="policy",
        status="faulted",
        fault_kind="replay_mismatch",
        detail="candidate_semantic_keys diverged",
    )


def _prune_node() -> StablePruneNodeTrace:
    return StablePruneNodeTrace(
        node_id="search:a",
        parent_node_id="search:root",
        branch_id="a",
        parent_branch_id="root",
        frontier_index_before_prune=0,
        kept=True,
        value=5.0,
        root_action_id="A",
        rng_id=1,
        decision_point_id="d-a",
        depth=1,
        combat_depth=1,
        continuation_steps=0,
        terminal=False,
        action_id="act-a",
        action_type="card",
        action={"action_id": "act-a", "action_type": "card"},
        policy_rank=0,
        policy_score=None,
        post_coverage_rank=0,
        candidate_source="policy",
    )


class OracleBranchFaultCensoringTest(unittest.TestCase):
    def test_terminal_sibling_does_not_hide_faulted_subtree(self) -> None:
        resolved_parent = _resolved(
            node_id="search:a",
            parent_node_id="search:root",
            branch_id="a",
            parent_branch_id="root",
            rng_id=1,
            value=5.0,
            terminal=False,
        )
        prune = StablePruneTrace(
            search_id="search",
            prune_step_id="search:prune:0",
            phase="stable_frontier",
            k=2,
            frontier_size=1,
            pruner_name="value_top_k",
            pruner_version="1",
            max_depth=3,
            depths_completed=0,
            remaining_time_ms=None,
            nodes=(_prune_node(),),
            selected_indices=(0,),
        )
        child_proposal = PolicyProposalTrace(
            search_id="search",
            proposal_step_id="search:proposal:1:0",
            parent_node_id="search:a",
            parent_branch_id="a",
            decision_point_id="d-a",
            legal_actions=(),
            candidates=(),
        )
        terminal_child = _resolved(
            node_id="search:a-ok",
            parent_node_id="search:a",
            branch_id="a-ok",
            parent_branch_id="a",
            rng_id=1,
            value=10.0,
            terminal=True,
        )
        faulted_child = _fault(
            branch_id="a-fault",
            parent_node_id="search:a",
            parent_branch_id="a",
            rng_id=1,
        )
        end = SearchTraceEnd(
            search_id="search",
            reason="beam_exhausted",
            best_root_action_id="A",
            best_value=10.0,
            depths_completed=2,
            nodes_expanded=2,
            branches_created=3,
            branches_faulted=1,
        )

        targets = build_oracle_targets(
            [
                _start(),
                _root_proposal(),
                resolved_parent,
                prune,
                child_proposal,
                terminal_child,
                faulted_child,
                end,
            ],
            target_beam_width=1,
            exhaustive_root_actions=True,
        )

        root = targets.root_actions[0]
        self.assertTrue(root.evaluated)
        self.assertIsNone(root.estimated_q)
        self.assertEqual(root.target_source, "no_target")
        self.assertTrue(root.censored)
        self.assertEqual(root.censor_reason, "branch_fault:replay_mismatch")
        self.assertEqual(len(root.rng_outcomes), 1)
        self.assertIsNone(root.rng_outcomes[0].value)
        self.assertEqual(root.rng_outcomes[0].target_source, "no_target")
        self.assertTrue(root.rng_outcomes[0].terminal_reached)
        self.assertEqual(root.rng_outcomes[0].deepest_combat_depth, 2)
        self.assertEqual(
            root.rng_outcomes[0].censor_reason,
            "branch_fault:replay_mismatch",
        )

        stable = targets.stable_nodes[0]
        self.assertIsNone(stable.target_value)
        self.assertEqual(stable.target_source, "no_target")
        self.assertTrue(stable.censored)
        self.assertTrue(stable.terminal_reached)
        self.assertEqual(stable.censor_reason, "branch_fault:replay_mismatch")
        self.assertIsNone(stable.best_descendant_node_id)

    def test_fault_only_rng_hypothesis_is_not_dropped_from_root_target(self) -> None:
        clean = _resolved(
            node_id="search:a-clean",
            parent_node_id="search:root",
            branch_id="a-clean",
            parent_branch_id="root",
            rng_id=1,
            value=10.0,
            terminal=True,
        )
        fault = _fault(
            branch_id="a-fault",
            parent_node_id="search:root",
            parent_branch_id="root",
            rng_id=2,
        )
        end = SearchTraceEnd(
            search_id="search",
            reason="beam_exhausted",
            best_root_action_id="A",
            best_value=10.0,
            depths_completed=1,
            nodes_expanded=1,
            branches_created=2,
            branches_faulted=1,
        )

        targets = build_oracle_targets(
            [_start(), _root_proposal(), clean, fault, end],
            target_beam_width=1,
            exhaustive_root_actions=True,
        )

        root = targets.root_actions[0]
        self.assertEqual([outcome.rng_id for outcome in root.rng_outcomes], [1, 2])
        self.assertEqual(root.rng_outcomes[0].value, 10.0)
        self.assertEqual(root.rng_outcomes[0].target_source, "terminal")
        self.assertIsNone(root.rng_outcomes[1].value)
        self.assertEqual(root.rng_outcomes[1].target_source, "no_target")
        self.assertEqual(root.rng_outcomes[1].deepest_combat_depth, 1)
        self.assertEqual(
            root.rng_outcomes[1].censor_reason,
            "branch_fault:replay_mismatch",
        )
        self.assertIsNone(root.estimated_q)
        self.assertEqual(root.target_source, "no_target")
        self.assertTrue(root.censored)
        self.assertEqual(root.censor_reason, "branch_fault:replay_mismatch")

    def test_all_faulted_root_action_is_censored_in_exhaustive_collection(self) -> None:
        faults = [
            _fault(
                branch_id="a-fault-1",
                parent_node_id="search:root",
                parent_branch_id="root",
                rng_id=1,
            ),
            _fault(
                branch_id="a-fault-2",
                parent_node_id="search:root",
                parent_branch_id="root",
                rng_id=2,
            ),
        ]
        end = SearchTraceEnd(
            search_id="search",
            reason="beam_exhausted",
            best_root_action_id=None,
            best_value=None,
            depths_completed=0,
            nodes_expanded=0,
            branches_created=2,
            branches_faulted=2,
        )

        targets = build_oracle_targets(
            [_start(), _root_proposal(), *faults, end],
            target_beam_width=1,
            exhaustive_root_actions=True,
        )

        root = targets.root_actions[0]
        self.assertTrue(root.evaluated)
        self.assertEqual([outcome.rng_id for outcome in root.rng_outcomes], [1, 2])
        self.assertTrue(all(outcome.censored for outcome in root.rng_outcomes))
        self.assertTrue(all(outcome.value is None for outcome in root.rng_outcomes))
        self.assertIsNone(root.estimated_q)
        self.assertEqual(root.target_source, "no_target")
        self.assertTrue(root.censored)
        self.assertEqual(root.censor_reason, "branch_fault:replay_mismatch")

    def test_all_faulted_root_action_is_evaluated_in_non_exhaustive_collection(self) -> None:
        fault = _fault(
            branch_id="a-fault",
            parent_node_id="search:root",
            parent_branch_id="root",
            rng_id=1,
        )
        end = SearchTraceEnd(
            search_id="search",
            reason="beam_exhausted",
            best_root_action_id=None,
            best_value=None,
            depths_completed=0,
            nodes_expanded=0,
            branches_created=1,
            branches_faulted=1,
        )

        targets = build_oracle_targets(
            [_start(), _root_proposal(), fault, end],
            target_beam_width=1,
            exhaustive_root_actions=False,
        )

        root = targets.root_actions[0]
        self.assertTrue(root.evaluated)
        self.assertEqual(len(root.rng_outcomes), 1)
        self.assertIsNone(root.rng_outcomes[0].value)
        self.assertEqual(root.rng_outcomes[0].censor_reason, "branch_fault:replay_mismatch")
        self.assertIsNone(root.estimated_q)
        self.assertEqual(root.target_source, "no_target")
        self.assertTrue(root.censored)
        self.assertEqual(root.censor_reason, "branch_fault:replay_mismatch")


if __name__ == "__main__":
    unittest.main()
