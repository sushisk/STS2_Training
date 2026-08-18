from __future__ import annotations

import unittest
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.decision.beam_search import (
    BeamSearchConfig,
    BeamSearchEngine,
    BranchFaultAbortError,
    BranchFaultPolicy,
)
from sts2_training.decision.policy import PriorHeuristicPolicy
from sts2_training.decision.value import HeuristicValueFunction


class _StructuralFaultClient:
    pending_retry = None
    session_invalid = False
    instance_type = "combat"
    max_emulate_actions_items = 64

    def __init__(self) -> None:
        self.emulate_calls = 0
        self.cancelled: list[tuple[str, ...]] = []
        self.released: list[tuple[str, ...]] = []

    async def emulate_actions(
        self,
        instance_id: str,
        items: Sequence[Mapping[str, Any]],
        *,
        timeout_s: float,
        simulation_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        del instance_id, timeout_s, simulation_options
        self.emulate_calls += 1
        return {
            "status": "completed",
            "branch_results": {
                item["branch_id"]: {
                    "status": "faulted",
                    "fault_kind": "worker_exception",
                    "error": (
                        "Snapshot restore rejected:\n"
                        "reference_integrity:Dangling reference 'creature-000027' "
                        "at power 'FRAIL_POWER'.ApplierInstanceId"
                    ),
                }
                for item in items
            },
        }

    async def cancel_branches(
        self,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        del instance_id, timeout_s
        self.cancelled.append(tuple(branch_ids))
        return {}

    async def release_branches(
        self,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        del instance_id, timeout_s
        self.released.append(tuple(branch_ids))
        return {}


class OracleBranchFaultSearchAbortTest(unittest.IsolatedAsyncioTestCase):
    async def test_structural_fault_aborts_search_and_still_cleans_created_branch(self) -> None:
        client = _StructuralFaultClient()
        engine = BeamSearchEngine(
            client,
            policy=PriorHeuristicPolicy(),
            value_fn=HeuristicValueFunction(),
            config=BeamSearchConfig(
                max_depth=1,
                top_k_actions=1,
                branch_fault_policy=BranchFaultPolicy(),
            ),
        )
        root_decision = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": {
                "legal_actions": [
                    {"action_id": "end", "action_type": "system", "is_available": True}
                ]
            },
        }

        with self.assertRaises(BranchFaultAbortError) as raised:
            await engine.search("inst-001", root_decision, timeout_s=1.0)

        exc = raised.exception
        self.assertEqual(exc.termination_reason, "aborted_snapshot_restore_fault")
        self.assertEqual(exc.summary.fault_signature, "snapshot_restore_fault")
        self.assertEqual(exc.summary.count, 1)
        self.assertEqual(exc.summary.first_depth, 1)
        self.assertEqual(exc.summary.action_ids, ("end",))
        self.assertEqual(exc.summary.action_types, ("system",))
        self.assertEqual(client.emulate_calls, 1)
        self.assertEqual(len(client.cancelled), 1)
        self.assertEqual(len(client.released), 1)
        self.assertEqual(client.cancelled[0], client.released[0])
        self.assertEqual(len(client.cancelled[0]), 1)


if __name__ == "__main__":
    unittest.main()
