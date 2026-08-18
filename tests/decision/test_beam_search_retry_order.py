import time
import unittest

from sts2_training.decision.beam_search import (
    BeamNode,
    BeamSearchConfig,
    BeamSearchEngine,
    BeamSearchStats,
)
from sts2_training.decision.candidate_coverage import CandidateProposal
from sts2_training.decision.policy import ActionCandidate


class _RetryOrderClient:
    max_emulate_actions_items = None

    def __init__(self) -> None:
        self.calls = 0

    async def emulate_actions(
        self,
        instance_id,
        items,
        *,
        timeout_s,
        simulation_options=None,
    ):
        del instance_id, timeout_s, simulation_options
        self.calls += 1
        branch_results = {}
        for item in items:
            if item["action_id"] == "first" and self.calls == 1:
                result = {
                    "status": "faulted",
                    "fault_kind": "worker_exception",
                    "error": "transient",
                }
            else:
                result = {
                    "status": "completed",
                    "decision_point_id": f"d-{item['action_id']}",
                    "masked_emulator_dto": {"legal_actions": []},
                }
            branch_results[item["branch_id"]] = result
        return {"status": "completed", "branch_results": branch_results}


class BeamSearchRetryOrderTests(unittest.IsolatedAsyncioTestCase):
    async def test_recovered_candidate_keeps_original_logical_frontier_order(self) -> None:
        client = _RetryOrderClient()
        engine = BeamSearchEngine(
            client,
            policy=object(),  # type: ignore[arg-type]
            value_fn=object(),  # type: ignore[arg-type]
            config=BeamSearchConfig(max_branch_attempts=2),
        )
        parent = BeamNode(
            branch_id="root",
            parent_branch_id="root",
            rng_id=0,
            decision_point_id="d-root",
            masked_emulator_dto={
                "legal_actions": [
                    {"action_id": "first", "action_type": "card", "is_available": True},
                    {"action_id": "second", "action_type": "card", "is_available": True},
                ]
            },
            depth=0,
            value=0.0,
            root_action_id=None,
        )
        first = CandidateProposal(
            candidate=ActionCandidate("first", 1.0),
            policy_rank=0,
            policy_score=1.0,
            post_coverage_rank=0,
            candidate_source="policy",
        )
        second = CandidateProposal(
            candidate=ActionCandidate("second", 1.0),
            policy_rank=1,
            policy_score=1.0,
            post_coverage_rank=1,
            candidate_source="policy",
        )
        items = [
            {
                "parent_branch_id": "root",
                "branch_id": "branch-first",
                "rng_id": 1,
                "decision_point_id": "d-root",
                "action_id": "first",
            },
            {
                "parent_branch_id": "root",
                "branch_id": "branch-second",
                "rng_id": 2,
                "decision_point_id": "d-root",
                "action_id": "second",
            },
        ]
        item_meta = [
            (parent, first, "branch-first", 1),
            (parent, second, "branch-second", 2),
        ]
        stats = BeamSearchStats()
        all_branch_ids = []

        branch_results, final_meta, fatal_reason = await engine._emulate_depth_batch(  # noqa: SLF001
            "inst",
            items,
            item_meta,
            all_branch_ids,
            stats,
            time.monotonic() + 5.0,
        )

        self.assertIsNone(fatal_reason)
        self.assertEqual([meta[1].action_id for meta in final_meta], ["first", "second"])
        self.assertNotEqual(final_meta[0][2], "branch-first")
        self.assertEqual(final_meta[1][2], "branch-second")
        self.assertEqual(set(branch_results), {final_meta[0][2], "branch-second"})
        self.assertEqual(stats.branches_created, 3)
        self.assertEqual(stats.branch_retry_faults, 1)
        self.assertEqual(stats.branch_retry_recoveries, 1)


if __name__ == "__main__":
    unittest.main()
