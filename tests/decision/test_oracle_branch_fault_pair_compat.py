from __future__ import annotations

import time
import unittest

from sts2_training.decision.beam_search import (
    BeamNode,
    BeamSearchConfig,
    BeamSearchEngine,
    BeamSearchStats,
    BranchFaultAbortError,
    BranchFaultPolicy,
)
from sts2_training.decision.branch_faults import (
    SNAPSHOT_RESTORE_FAULT_SIGNATURE,
    classify_known_branch_fault,
)
from sts2_training.decision.candidate_coverage import CandidateProposal
from sts2_training.decision.policy import ActionCandidate


_SETTLEMENT_TIMEOUT = "Timed out waiting for the next decision point or settlement"


class _SequenceClient:
    max_emulate_actions_items = None

    def __init__(self, results: list[dict]) -> None:
        self.results = results
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
        result_template = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return {
            "status": "completed",
            "branch_results": {
                item["branch_id"]: dict(result_template) for item in items
            },
        }


def _item_and_meta():
    action_id = "end"
    parent = BeamNode(
        branch_id="root",
        parent_branch_id="root",
        rng_id=0,
        decision_point_id="d-root",
        masked_emulator_dto={
            "legal_actions": [
                {"action_id": action_id, "action_type": "system", "is_available": True}
            ]
        },
        depth=0,
        value=0.0,
        root_action_id=None,
    )
    candidate = CandidateProposal(
        candidate=ActionCandidate(action_id, 1.0),
        policy_rank=0,
        policy_score=1.0,
        post_coverage_rank=0,
        candidate_source="policy",
    )
    branch_id = "branch-end"
    item = {
        "parent_branch_id": "root",
        "branch_id": branch_id,
        "rng_id": 1,
        "decision_point_id": "d-root",
        "action_id": action_id,
    }
    return [item], [(parent, candidate, branch_id, 1)]


def _engine(client: _SequenceClient) -> BeamSearchEngine:
    return BeamSearchEngine(
        client,
        policy=object(),  # type: ignore[arg-type]
        value_fn=object(),  # type: ignore[arg-type]
        config=BeamSearchConfig(
            max_branch_attempts=3,
            branch_fault_policy=BranchFaultPolicy(),
        ),
    )


class OracleBranchFaultPairCompatibilityTest(unittest.IsolatedAsyncioTestCase):
    async def test_current_rl_missing_monster_move_fault_is_structural_and_not_retried(self) -> None:
        result = {
            "status": "faulted",
            "fault_kind": "snapshot_restore_missing_monster_move",
            "error": (
                "Refusing to execute End Turn: living enemies still lack a current Move "
                "after snapshot restore; post-restore invariant violation"
            ),
        }
        client = _SequenceClient([result])
        engine = _engine(client)
        items, meta = _item_and_meta()
        stats = BeamSearchStats()

        with self.assertRaises(BranchFaultAbortError) as raised:
            await engine._emulate_depth_batch(  # noqa: SLF001
                "inst",
                items,
                meta,
                [],
                stats,
                time.monotonic() + 5.0,
                search_id="search-current-rl-restore-invariant",
            )

        self.assertEqual(client.calls, 1)
        self.assertEqual(stats.branch_retry_faults, 0)
        self.assertEqual(raised.exception.termination_reason, "aborted_snapshot_restore_fault")
        self.assertEqual(
            raised.exception.summary.fault_signature,
            SNAPSHOT_RESTORE_FAULT_SIGNATURE,
        )
        self.assertEqual(
            raised.exception.summary.fault_kind,
            "snapshot_restore_missing_monster_move",
        )

    async def test_first_timeout_after_generic_fault_still_gets_timeout_retry(self) -> None:
        client = _SequenceClient(
            [
                {
                    "status": "faulted",
                    "fault_kind": "worker_exception",
                    "error": "generic transient failure",
                },
                {
                    "status": "faulted",
                    "fault_kind": "worker_exception",
                    "error": _SETTLEMENT_TIMEOUT,
                },
                {
                    "status": "completed",
                    "decision_point_id": "d-next",
                    "masked_emulator_dto": {"terminal": True, "legal_actions": []},
                },
            ]
        )
        engine = _engine(client)
        items, meta = _item_and_meta()
        stats = BeamSearchStats()

        branch_results, final_meta, fatal_reason = await engine._emulate_depth_batch(  # noqa: SLF001
            "inst",
            items,
            meta,
            [],
            stats,
            time.monotonic() + 5.0,
            search_id="search-mixed-fault-sequence",
        )

        self.assertIsNone(fatal_reason)
        self.assertEqual(client.calls, 3)
        self.assertEqual(stats.branch_retry_faults, 2)
        self.assertEqual(stats.branch_retry_recoveries, 1)
        self.assertEqual(len(final_meta), 1)
        self.assertEqual(branch_results[final_meta[0][2]]["status"], "completed")

    async def test_first_timeout_at_global_attempt_limit_is_not_persistent_timeout(self) -> None:
        client = _SequenceClient(
            [
                {
                    "status": "faulted",
                    "fault_kind": "worker_exception",
                    "error": "generic transient failure 1",
                },
                {
                    "status": "faulted",
                    "fault_kind": "worker_exception",
                    "error": "generic transient failure 2",
                },
                {
                    "status": "faulted",
                    "fault_kind": "worker_exception",
                    "error": _SETTLEMENT_TIMEOUT,
                },
            ]
        )
        engine = _engine(client)
        items, meta = _item_and_meta()
        stats = BeamSearchStats()

        branch_results, final_meta, fatal_reason = await engine._emulate_depth_batch(  # noqa: SLF001
            "inst",
            items,
            meta,
            [],
            stats,
            time.monotonic() + 5.0,
            search_id="search-timeout-at-global-limit",
        )

        self.assertIsNone(fatal_reason)
        self.assertEqual(client.calls, 3)
        self.assertEqual(stats.branch_retry_faults, 2)
        self.assertEqual(stats.branch_retry_recoveries, 0)
        self.assertEqual(len(final_meta), 1)
        final_result = branch_results[final_meta[0][2]]
        self.assertEqual(final_result["status"], "faulted")
        self.assertEqual(final_result["error"], _SETTLEMENT_TIMEOUT)

    def test_shared_classifier_matches_current_rl_typed_restore_invariant(self) -> None:
        signature = classify_known_branch_fault(
            status="faulted",
            fault_kind="snapshot_restore_missing_monster_move",
            detail="Refusing to execute End Turn because restored monster Move is missing",
        )

        self.assertEqual(signature, SNAPSHOT_RESTORE_FAULT_SIGNATURE)

    def test_worker_task_timeout_is_not_settlement_timeout(self) -> None:
        signature = classify_known_branch_fault(
            status="faulted",
            fault_kind="task_timeout",
            detail="timed out waiting for worker request 123 after 20 seconds",
        )

        self.assertIsNone(signature)


if __name__ == "__main__":
    unittest.main()
