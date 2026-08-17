from __future__ import annotations

from sts2_training.decision.oracle_search import build_oracle_targets
from sts2_training.decision.search_trace import (
    BranchFaultTrace,
    PolicyCandidateTrace,
    PolicyProposalTrace,
    ResolvedNodeTrace,
    SearchTraceEnd,
    SearchTraceStart,
)


def test_non_replay_branch_fault_censors_root_target() -> None:
    """Target safety is driven by branch failure, not replay-provenance semantics."""

    action = {"action_id": "A", "action_type": "card", "is_available": True}
    start = SearchTraceStart(
        search_id="search",
        instance_id="inst",
        root_decision_point_id="d-root",
        beam_width=2,
        top_k_actions=2,
        max_depth=1,
        max_continuation_steps=8,
        time_budget_ms=None,
        pruner_name="value_top_k",
        pruner_version="1",
    )
    proposal = PolicyProposalTrace(
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
                branch_id="a-clean",
                rng_id=1,
                policy_rank=0,
                policy_score=None,
                post_coverage_rank=0,
                candidate_source="policy",
            ),
            PolicyCandidateTrace(
                action_id="A",
                action=action,
                branch_id="a-timeout",
                rng_id=2,
                policy_rank=1,
                policy_score=None,
                post_coverage_rank=1,
                candidate_source="policy",
            ),
        ),
    )
    clean = ResolvedNodeTrace(
        search_id="search",
        node_id="search:a-clean",
        parent_node_id="search:root",
        branch_id="a-clean",
        parent_branch_id="root",
        root_action_id="A",
        rng_id=1,
        decision_point_id="d-clean",
        depth=1,
        combat_depth=1,
        continuation_steps=0,
        value=10.0,
        value_is_fresh=True,
        value_source="terminal",
        state_kind="terminal",
        resolution="terminal",
        terminal=True,
        action_id="A",
        action_type="card",
        action=action,
        policy_rank=0,
        policy_score=None,
        post_coverage_rank=0,
        candidate_source="policy",
    )
    timeout = BranchFaultTrace(
        search_id="search",
        node_id="search:a-timeout",
        parent_node_id="search:root",
        branch_id="a-timeout",
        parent_branch_id="root",
        root_action_id="A",
        rng_id=2,
        depth=1,
        combat_depth=1,
        continuation_steps=0,
        action_id="A",
        action_type="card",
        action=action,
        policy_rank=1,
        policy_score=None,
        post_coverage_rank=1,
        candidate_source="policy",
        status="faulted",
        fault_kind="task_timeout",
        detail="branch execution exceeded its deadline",
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
        [start, proposal, clean, timeout, end],
        target_beam_width=1,
        exhaustive_root_actions=True,
    )

    root = targets.root_actions[0]
    assert root.evaluated is True
    assert [outcome.rng_id for outcome in root.rng_outcomes] == [1, 2]
    assert root.rng_outcomes[0].value == 10.0
    assert root.rng_outcomes[0].target_source == "terminal"
    assert root.rng_outcomes[1].value is None
    assert root.rng_outcomes[1].target_source == "no_target"
    assert root.rng_outcomes[1].censor_reason == "branch_fault:task_timeout"
    assert root.estimated_q is None
    assert root.target_source == "no_target"
    assert root.censored is True
    assert root.censor_reason == "branch_fault:task_timeout"
