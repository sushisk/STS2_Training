from __future__ import annotations

import unittest

from sts2_training.decision.oracle_log import _serialized_search_trace
from sts2_training.decision.search_trace import BranchFaultTrace, SearchTraceEnd


def _fault(
    *,
    branch_id: str,
    root_action_id: str,
    action_id: str,
    action_type: str,
    depth: int,
    detail: str,
) -> BranchFaultTrace:
    return BranchFaultTrace(
        search_id="search",
        node_id=f"search:{branch_id}",
        parent_node_id="search:parent",
        branch_id=branch_id,
        parent_branch_id="parent",
        root_action_id=root_action_id,
        rng_id=1,
        depth=depth,
        combat_depth=depth,
        continuation_steps=0,
        action_id=action_id,
        action_type=action_type,
        action={"action_id": action_id, "action_type": action_type},
        policy_rank=0,
        policy_score=None,
        post_coverage_rank=0,
        candidate_source="policy",
        status="faulted",
        fault_kind="worker_exception",
        detail=detail,
    )


class OracleFaultLogAggregationTest(unittest.TestCase):
    def test_repeated_fault_detail_is_kept_once_and_search_end_gets_summary(self) -> None:
        detail = "Timed out waiting for the next decision point or settlement\nfull stack"
        first = _fault(
            branch_id="b-1",
            root_action_id="root-a",
            action_id="end-a",
            action_type="system",
            depth=2,
            detail=detail,
        )
        second = _fault(
            branch_id="b-2",
            root_action_id="root-b",
            action_id="end-b",
            action_type="system",
            depth=4,
            detail=detail,
        )
        end = SearchTraceEnd(
            search_id="search",
            reason="max_depth",
            best_root_action_id="root-ok",
            best_value=1.0,
            depths_completed=4,
            nodes_expanded=5,
            branches_created=7,
            branches_faulted=2,
        )

        payloads = _serialized_search_trace((first, second, end))

        self.assertEqual(payloads[0]["detail"], detail)
        self.assertIsNone(payloads[1]["detail"])
        summaries = payloads[2]["fault_summaries"]
        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertEqual(summary["fault_signature"], "settlement_timeout")
        self.assertEqual(summary["fault_kind"], "worker_exception")
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["first_detail"], detail)
        self.assertEqual(summary["first_depth"], 2)
        self.assertEqual(summary["last_depth"], 4)
        self.assertEqual(summary["first_branch_id"], "b-1")
        self.assertEqual(summary["last_branch_id"], "b-2")
        self.assertEqual(summary["action_ids"], ["end-a", "end-b"])
        self.assertEqual(summary["action_types"], ["system"])
        self.assertEqual(summary["root_action_ids"], ["root-a", "root-b"])

        # Persistence filtering must not mutate the in-memory trace used to derive targets.
        self.assertEqual(first.detail, detail)
        self.assertEqual(second.detail, detail)
        self.assertEqual(end.fault_summaries, ())

    def test_structural_signature_collapses_reference_integrity_variants(self) -> None:
        first = _fault(
            branch_id="struct-1",
            root_action_id="root-a",
            action_id="a",
            action_type="card",
            depth=1,
            detail=(
                "Snapshot restore rejected: reference_integrity:Dangling reference "
                "'creature-000027'"
            ),
        )
        second = _fault(
            branch_id="struct-2",
            root_action_id="root-b",
            action_id="b",
            action_type="card",
            depth=3,
            detail=(
                "Snapshot restore rejected: reference_integrity:Dangling reference "
                "'creature-999999'"
            ),
        )
        end = SearchTraceEnd(
            search_id="search",
            reason="beam_exhausted",
            best_root_action_id=None,
            best_value=None,
            depths_completed=3,
            nodes_expanded=0,
            branches_created=2,
            branches_faulted=2,
        )

        payloads = _serialized_search_trace((first, second, end))

        self.assertIsNone(payloads[1]["detail"])
        summaries = payloads[2]["fault_summaries"]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["fault_signature"], "snapshot_restore_fault")
        self.assertEqual(summaries[0]["count"], 2)
        self.assertEqual(summaries[0]["first_depth"], 1)
        self.assertEqual(summaries[0]["last_depth"], 3)


if __name__ == "__main__":
    unittest.main()
