from __future__ import annotations

import unittest

from sts2_training.decision.oracle_search import build_oracle_targets
from sts2_training.decision.search_trace import (
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
        beam_width=1,
        top_k_actions=1,
        max_depth=2,
        max_continuation_steps=8,
        time_budget_ms=None,
        pruner_name="value_top_k",
        pruner_version="1",
    )


def _proposal() -> PolicyProposalTrace:
    action = {"action_id": "A", "action_type": "card", "is_available": True}
    return PolicyProposalTrace(
        search_id="search",
        proposal_step_id="p",
        parent_node_id="search:root",
        parent_branch_id="root",
        decision_point_id="d-root",
        legal_actions=(action,),
        candidates=(
            PolicyCandidateTrace("A", action, "a", 1, 0, None, 0, "policy"),
        ),
    )


def _resolved_root(*, value: float = 99.0) -> ResolvedNodeTrace:
    return ResolvedNodeTrace(
        search_id="search",
        node_id="search:a",
        parent_node_id="search:root",
        branch_id="a",
        parent_branch_id="root",
        root_action_id="A",
        rng_id=1,
        decision_point_id="d-a",
        depth=1,
        combat_depth=1,
        continuation_steps=0,
        value=value,
        value_is_fresh=True,
        value_source="value_bootstrap",
        state_kind="stable",
        resolution="expandable_stable",
        terminal=False,
        action_id="A",
        action_type="card",
        action={"action_id": "A", "action_type": "card"},
        policy_rank=0,
        policy_score=None,
        post_coverage_rank=0,
        candidate_source="policy",
    )


def _pruned_root(*, value: float = 99.0) -> StablePruneTrace:
    return StablePruneTrace(
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
            StablePruneNodeTrace(
                node_id="search:a",
                parent_node_id="search:root",
                branch_id="a",
                parent_branch_id="root",
                frontier_index_before_prune=0,
                kept=False,
                value=value,
                root_action_id="A",
                rng_id=1,
                decision_point_id="d-a",
                depth=1,
                combat_depth=1,
                continuation_steps=0,
                terminal=False,
                action_id="A",
                action_type="card",
                action={"action_id": "A", "action_type": "card"},
                policy_rank=0,
                policy_score=None,
                post_coverage_rank=0,
                candidate_source="policy",
            ),
        ),
    )


class OracleReviewRegressionTest(unittest.TestCase):
    def test_pruned_root_node_is_not_reused_as_root_q_target(self) -> None:
        targets = build_oracle_targets(
            [
                _start(),
                _proposal(),
                _resolved_root(),
                _pruned_root(),
                SearchTraceEnd("search", "beam_exhausted", None, None, 1, 1, 1),
            ],
            target_beam_width=1,
            exhaustive_root_actions=True,
        )

        root = targets.root_actions[0]
        self.assertTrue(root.evaluated)
        self.assertIsNone(root.estimated_q)
        self.assertEqual(root.target_source, "no_target")
        self.assertEqual(root.censor_reason, "oracle_pruned_before_followup")
        self.assertEqual(len(root.rng_outcomes), 1)
        self.assertIsNone(root.rng_outcomes[0].value)
        self.assertEqual(
            root.rng_outcomes[0].censor_reason,
            "oracle_pruned_before_followup",
        )

    def test_unmaterialized_policy_candidate_is_not_marked_evaluated(self) -> None:
        targets = build_oracle_targets(
            [
                _start(),
                _proposal(),
                SearchTraceEnd(
                    "search",
                    "active_branch_capacity",
                    None,
                    None,
                    0,
                    0,
                    0,
                ),
            ],
            target_beam_width=1,
            exhaustive_root_actions=False,
        )

        root = targets.root_actions[0]
        self.assertFalse(root.evaluated)
        self.assertIsNone(root.estimated_q)
        self.assertEqual(root.rng_outcomes, ())
        self.assertEqual(root.target_source, "no_target")
        self.assertEqual(
            root.censor_reason,
            "candidate_not_materialized:active_branch_capacity",
        )

    def test_exhaustive_root_fails_closed_if_candidate_is_not_materialized(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "materialize at least one resolved outcome"):
            build_oracle_targets(
                [
                    _start(),
                    _proposal(),
                    SearchTraceEnd(
                        "search",
                        "emulate_actions_rejected:capacity",
                        None,
                        None,
                        0,
                        0,
                        0,
                    ),
                ],
                target_beam_width=1,
                exhaustive_root_actions=True,
            )


if __name__ == "__main__":
    unittest.main()
