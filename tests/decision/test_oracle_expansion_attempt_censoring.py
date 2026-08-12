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
        beam_width=2,
        top_k_actions=2,
        max_depth=3,
        max_continuation_steps=8,
        time_budget_ms=None,
        pruner_name="value_top_k",
        pruner_version="1",
    )


def _action(action_id: str) -> dict[str, object]:
    return {
        "action_id": action_id,
        "action_type": "card",
        "is_available": True,
    }


def _proposal(
    *,
    proposal_step_id: str,
    parent_node_id: str,
    parent_branch_id: str,
    decision_point_id: str,
    action_id: str,
    branch_id: str,
    rng_id: int,
) -> PolicyProposalTrace:
    action = _action(action_id)
    return PolicyProposalTrace(
        search_id="search",
        proposal_step_id=proposal_step_id,
        parent_node_id=parent_node_id,
        parent_branch_id=parent_branch_id,
        decision_point_id=decision_point_id,
        legal_actions=(action,),
        candidates=(
            PolicyCandidateTrace(
                action_id,
                action,
                branch_id,
                rng_id,
                0,
                None,
                0,
                "policy",
            ),
        ),
    )


def _resolved(
    *,
    node_id: str,
    parent_node_id: str,
    branch_id: str,
    parent_branch_id: str,
    decision_point_id: str,
    action_id: str,
    value: float,
    depth: int,
) -> ResolvedNodeTrace:
    return ResolvedNodeTrace(
        search_id="search",
        node_id=node_id,
        parent_node_id=parent_node_id,
        branch_id=branch_id,
        parent_branch_id=parent_branch_id,
        root_action_id="A",
        rng_id=1,
        decision_point_id=decision_point_id,
        depth=depth,
        combat_depth=depth,
        continuation_steps=0,
        value=value,
        value_is_fresh=True,
        value_source="value_bootstrap",
        state_kind="stable",
        resolution="expandable_stable",
        terminal=False,
        action_id=action_id,
        action_type="card",
        action=_action(action_id),
        policy_rank=0,
        policy_score=None,
        post_coverage_rank=0,
        candidate_source="policy",
    )


def _prune(
    *,
    prune_step_id: str,
    node_id: str,
    parent_node_id: str,
    branch_id: str,
    parent_branch_id: str,
    decision_point_id: str,
    action_id: str,
    value: float,
    depth: int,
) -> StablePruneTrace:
    return StablePruneTrace(
        search_id="search",
        prune_step_id=prune_step_id,
        phase="stable_frontier",
        k=2,
        frontier_size=1,
        pruner_name="value_top_k",
        pruner_version="1",
        max_depth=3,
        depths_completed=depth - 1,
        remaining_time_ms=None,
        nodes=(
            StablePruneNodeTrace(
                node_id=node_id,
                parent_node_id=parent_node_id,
                branch_id=branch_id,
                parent_branch_id=parent_branch_id,
                frontier_index_before_prune=0,
                kept=True,
                value=value,
                root_action_id="A",
                rng_id=1,
                decision_point_id=decision_point_id,
                depth=depth,
                combat_depth=depth,
                continuation_steps=0,
                terminal=False,
                action_id=action_id,
                action_type="card",
                action=_action(action_id),
                policy_rank=0,
                policy_score=None,
                post_coverage_rank=0,
                candidate_source="policy",
            ),
        ),
    )


class OracleExpansionAttemptCensoringTest(unittest.TestCase):
    def test_failed_descendant_expansion_is_not_reused_as_root_or_stable_target(self) -> None:
        events = [
            _start(),
            _proposal(
                proposal_step_id="root-proposal",
                parent_node_id="search:root",
                parent_branch_id="root",
                decision_point_id="d-root",
                action_id="A",
                branch_id="a",
                rng_id=1,
            ),
            _resolved(
                node_id="search:a",
                parent_node_id="search:root",
                branch_id="a",
                parent_branch_id="root",
                decision_point_id="d-a",
                action_id="A",
                value=5.0,
                depth=1,
            ),
            _prune(
                prune_step_id="prune-a",
                node_id="search:a",
                parent_node_id="search:root",
                branch_id="a",
                parent_branch_id="root",
                decision_point_id="d-a",
                action_id="A",
                value=5.0,
                depth=1,
            ),
            _proposal(
                proposal_step_id="proposal-a",
                parent_node_id="search:a",
                parent_branch_id="a",
                decision_point_id="d-a",
                action_id="B",
                branch_id="b",
                rng_id=1,
            ),
            _resolved(
                node_id="search:b",
                parent_node_id="search:a",
                branch_id="b",
                parent_branch_id="a",
                decision_point_id="d-b",
                action_id="B",
                value=9.0,
                depth=2,
            ),
            _prune(
                prune_step_id="prune-b",
                node_id="search:b",
                parent_node_id="search:a",
                branch_id="b",
                parent_branch_id="a",
                decision_point_id="d-b",
                action_id="B",
                value=9.0,
                depth=2,
            ),
            # The proposal proves B was selected for expansion, but no resolved C exists.
            _proposal(
                proposal_step_id="proposal-b",
                parent_node_id="search:b",
                parent_branch_id="b",
                decision_point_id="d-b",
                action_id="C",
                branch_id="c",
                rng_id=1,
            ),
            SearchTraceEnd(
                "search",
                "emulate_actions_rejected:test",
                None,
                None,
                2,
                2,
                3,
            ),
        ]

        targets = build_oracle_targets(
            events,
            target_beam_width=1,
            exhaustive_root_actions=True,
        )

        root = targets.root_actions[0]
        self.assertTrue(root.evaluated)
        self.assertIsNone(root.estimated_q)
        self.assertEqual(root.target_source, "no_target")
        self.assertEqual(
            root.censor_reason,
            "expansion_attempted_without_outcome:emulate_actions_rejected:test",
        )
        self.assertEqual(len(root.rng_outcomes), 1)
        self.assertIsNone(root.rng_outcomes[0].value)

        ancestor = next(
            target for target in targets.stable_nodes if target.prune_step_id == "prune-a"
        )
        self.assertIsNone(ancestor.target_value)
        self.assertEqual(ancestor.target_source, "no_target")
        self.assertEqual(
            ancestor.censor_reason,
            "oracle_descendant_expansion_without_outcome:emulate_actions_rejected:test",
        )


if __name__ == "__main__":
    unittest.main()
