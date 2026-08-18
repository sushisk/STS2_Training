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
from sts2_training.decision.candidate_coverage import CandidateProposal
from sts2_training.decision.policy import ActionCandidate


class _FaultPolicyClient:
    max_emulate_actions_items = None

    def __init__(self, *, timeout_recovers: bool = False) -> None:
        self.calls = 0
        self.timeout_recovers = timeout_recovers

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
            action_id = item["action_id"]
            if action_id == "structural":
                result = {
                    "status": "faulted",
                    "fault_kind": "worker_exception",
                    "error": (
                        "Snapshot restore rejected:\n"
                        "reference_integrity:Dangling reference 'creature-000027' "
                        "at power 'FRAIL_POWER'.ApplierInstanceId"
                    ),
                }
            elif action_id == "timeout" and not (
                self.timeout_recovers and self.calls > 1
            ):
                result = {
                    "status": "faulted",
                    "fault_kind": "worker_exception",
                    "error": "Timed out waiting for the next decision point or settlement",
                }
            elif action_id == "generic":
                result = {
                    "status": "faulted",
                    "fault_kind": "worker_exception",
                    "error": "generic transient failure",
                }
            else:
                result = {
                    "status": "completed",
                    "decision_point_id": f"d-{action_id}",
                    "masked_emulator_dto": {"terminal": True, "legal_actions": []},
                }
            branch_results[item["branch_id"]] = result
        return {"status": "completed", "branch_results": branch_results}


def _frontier(action_ids: list[str]):
    parent = BeamNode(
        branch_id="root",
        parent_branch_id="root",
        rng_id=0,
        decision_point_id="d-root",
        masked_emulator_dto={
            "legal_actions": [
                {"action_id": action_id, "action_type": "card", "is_available": True}
                for action_id in action_ids
            ]
        },
        depth=0,
        value=0.0,
        root_action_id=None,
    )
    items = []
    meta = []
    for index, action_id in enumerate(action_ids, start=1):
        candidate = CandidateProposal(
            candidate=ActionCandidate(action_id, 1.0),
            policy_rank=index - 1,
            policy_score=1.0,
            post_coverage_rank=index - 1,
            candidate_source="policy",
        )
        branch_id = f"branch-{action_id}-{index}"
        items.append(
            {
                "parent_branch_id": "root",
                "branch_id": branch_id,
                "rng_id": index,
                "decision_point_id": "d-root",
                "action_id": action_id,
            }
        )
        meta.append((parent, candidate, branch_id, index))
    return items, meta


class OracleBranchFaultPolicyTests(unittest.IsolatedAsyncioTestCase):
    def _engine(self, client, *, policy: BranchFaultPolicy) -> BeamSearchEngine:
        return BeamSearchEngine(
            client,
            policy=object(),  # type: ignore[arg-type]
            value_fn=object(),  # type: ignore[arg-type]
            config=BeamSearchConfig(
                max_branch_attempts=3,
                branch_fault_policy=policy,
            ),
        )

    async def test_structural_fault_aborts_without_retry_and_keeps_one_summary(self) -> None:
        client = _FaultPolicyClient()
        engine = self._engine(client, policy=BranchFaultPolicy())
        items, meta = _frontier(["structural", "ok"])
        stats = BeamSearchStats()
        all_branch_ids: list[str] = []

        with self.assertRaises(BranchFaultAbortError) as raised:
            await engine._emulate_depth_batch(  # noqa: SLF001
                "inst",
                items,
                meta,
                all_branch_ids,
                stats,
                time.monotonic() + 5.0,
                search_id="search-structural",
            )

        exc = raised.exception
        self.assertEqual(client.calls, 1)
        self.assertEqual(exc.termination_reason, "aborted_snapshot_restore_fault")
        self.assertEqual(exc.summary.search_id, "search-structural")
        self.assertEqual(exc.summary.fault_signature, "snapshot_restore_fault")
        self.assertEqual(exc.summary.count, 1)
        self.assertEqual(exc.summary.first_depth, 1)
        self.assertEqual(exc.summary.last_depth, 1)
        self.assertEqual(exc.summary.action_ids, ("structural",))
        self.assertEqual(exc.summary.action_types, ("card",))
        self.assertEqual(exc.summary.root_action_ids, ("structural",))
        self.assertIn("reference_integrity:", exc.summary.first_detail or "")
        self.assertEqual(stats.branches_created, 2)
        self.assertEqual(stats.branch_retry_faults, 0)

    async def test_settlement_timeout_gets_one_retry_and_can_recover(self) -> None:
        client = _FaultPolicyClient(timeout_recovers=True)
        engine = self._engine(client, policy=BranchFaultPolicy())
        items, meta = _frontier(["timeout", "ok"])
        stats = BeamSearchStats()
        all_branch_ids: list[str] = []

        branch_results, final_meta, fatal_reason = await engine._emulate_depth_batch(  # noqa: SLF001
            "inst",
            items,
            meta,
            all_branch_ids,
            stats,
            time.monotonic() + 5.0,
            search_id="search-timeout-recover",
        )

        self.assertIsNone(fatal_reason)
        self.assertEqual(client.calls, 2)
        self.assertEqual([entry[1].action_id for entry in final_meta], ["timeout", "ok"])
        self.assertEqual(len(branch_results), 2)
        self.assertEqual(stats.branches_created, 3)
        self.assertEqual(stats.branch_retry_faults, 1)
        self.assertEqual(stats.branch_retry_recoveries, 1)

    async def test_persistent_timeout_stops_after_two_attempts_below_abort_budget(self) -> None:
        client = _FaultPolicyClient()
        engine = self._engine(
            client,
            policy=BranchFaultPolicy(
                settlement_timeout_abort_count=3,
                settlement_timeout_abort_ratio=0.50,
            ),
        )
        items, meta = _frontier(["timeout", "ok-1", "ok-2", "ok-3"])
        stats = BeamSearchStats()
        all_branch_ids: list[str] = []

        branch_results, final_meta, fatal_reason = await engine._emulate_depth_batch(  # noqa: SLF001
            "inst",
            items,
            meta,
            all_branch_ids,
            stats,
            time.monotonic() + 5.0,
            search_id="search-timeout-below-budget",
        )

        self.assertIsNone(fatal_reason)
        self.assertEqual(client.calls, 2)
        self.assertEqual(
            [entry[1].action_id for entry in final_meta],
            ["timeout", "ok-1", "ok-2", "ok-3"],
        )
        timeout_meta = final_meta[0]
        self.assertEqual(branch_results[timeout_meta[2]]["status"], "faulted")
        self.assertEqual(stats.branches_created, 5)
        self.assertEqual(stats.branch_retry_faults, 1)
        self.assertEqual(stats.branch_retry_recoveries, 0)

    async def test_persistent_timeout_aborts_after_retry_when_ratio_reaches_budget(self) -> None:
        client = _FaultPolicyClient()
        engine = self._engine(
            client,
            policy=BranchFaultPolicy(
                settlement_timeout_abort_count=99,
                settlement_timeout_abort_ratio=0.50,
            ),
        )
        items, meta = _frontier(["timeout", "ok"])
        stats = BeamSearchStats()
        all_branch_ids: list[str] = []

        with self.assertRaises(BranchFaultAbortError) as raised:
            await engine._emulate_depth_batch(  # noqa: SLF001
                "inst",
                items,
                meta,
                all_branch_ids,
                stats,
                time.monotonic() + 5.0,
                search_id="search-timeout-budget",
            )

        exc = raised.exception
        self.assertEqual(client.calls, 2)
        self.assertEqual(exc.termination_reason, "aborted_settlement_timeout_budget")
        self.assertEqual(exc.summary.fault_signature, "settlement_timeout")
        self.assertEqual(exc.summary.count, 1)
        self.assertEqual(exc.summary.action_ids, ("timeout",))
        self.assertEqual(stats.branches_created, 3)
        self.assertEqual(stats.branch_retry_faults, 1)

    async def test_generic_fault_keeps_existing_max_branch_attempts(self) -> None:
        client = _FaultPolicyClient()
        engine = self._engine(client, policy=BranchFaultPolicy())
        items, meta = _frontier(["generic", "ok"])
        stats = BeamSearchStats()
        all_branch_ids: list[str] = []

        branch_results, final_meta, fatal_reason = await engine._emulate_depth_batch(  # noqa: SLF001
            "inst",
            items,
            meta,
            all_branch_ids,
            stats,
            time.monotonic() + 5.0,
            search_id="search-generic",
        )

        self.assertIsNone(fatal_reason)
        self.assertEqual(client.calls, 3)
        self.assertEqual([entry[1].action_id for entry in final_meta], ["generic", "ok"])
        self.assertEqual(branch_results[final_meta[0][2]]["status"], "faulted")
        self.assertEqual(stats.branches_created, 4)
        self.assertEqual(stats.branch_retry_faults, 2)
        self.assertEqual(stats.branch_retry_recoveries, 0)


if __name__ == "__main__":
    unittest.main()
