from __future__ import annotations

import unittest

import sts2_training.decision as decision_api
from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.decision.oracle_search import BudgetedOracleCollector, OracleCollectionConfig
from sts2_training.decision.policy import PriorHeuristicPolicy
from sts2_training.decision.search_trace import BranchFaultTrace, SearchTraceEnd
from sts2_training.decision.value import HeuristicValueFunction


_ACTIONS = [
    {"action_id": "strike", "action_type": "card", "is_available": True},
    {"action_id": "end", "action_type": "system", "is_available": True},
]


class _PartialBranchResultClient:
    instance_type = "combat"
    max_emulate_actions_items = 64

    async def emulate_actions(
        self,
        instance_id: str,
        items: list[dict],
        *,
        timeout_s: float,
        simulation_options: dict | None = None,
    ) -> dict:
        del instance_id, timeout_s, simulation_options
        # The request itself succeeds, so every item in the chunk was admitted/emulated
        # from Training's perspective. Deliberately omit only the strike branch result.
        branch_results: dict[str, dict] = {}
        for item in items:
            if item["action_id"] != "end":
                continue
            branch_results[item["branch_id"]] = {
                "status": "completed",
                "decision_point_id": "d-end",
                "masked_emulator_dto": {
                    "terminal": True,
                    "outcome": "victory",
                    "hp": 40,
                    "maxHp": 50,
                    "enemies": [],
                    "legal_actions": [],
                },
                "branch_log": [],
            }
        return {"branch_results": branch_results}




class _AllMissingBranchResultClient(_PartialBranchResultClient):
    async def emulate_actions(
        self,
        instance_id: str,
        items: list[dict],
        *,
        timeout_s: float,
        simulation_options: dict | None = None,
    ) -> dict:
        del instance_id, items, timeout_s, simulation_options
        return {"branch_results": {}}


class OracleMissingBranchResultTest(unittest.IsolatedAsyncioTestCase):
    def test_branch_fault_trace_is_part_of_public_decision_api(self) -> None:
        self.assertIs(decision_api.BranchFaultTrace, BranchFaultTrace)
        self.assertIn("BranchFaultTrace", decision_api.__all__)

    async def test_missing_result_is_faulted_counted_and_censored(self) -> None:
        collector = BudgetedOracleCollector(
            _PartialBranchResultClient(),
            policy=PriorHeuristicPolicy(),
            value_fn=HeuristicValueFunction(),
            config=OracleCollectionConfig(
                beam_config=BeamSearchConfig(
                    beam_width=2,
                    top_k_actions=2,
                    max_depth=1,
                    release_branches_on_finish=False,
                ),
                target_beam_width=2,
                exhaustive_root_actions=True,
            ),
        )
        root_decision = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": {
                "hp": 40,
                "maxHp": 50,
                "enemies": [
                    {"hp": 10, "maxHp": 10, "isAlive": True, "intent": {}}
                ],
                "legal_actions": _ACTIONS,
            },
        }

        collected = await collector.collect("inst-001", root_decision, timeout_s=1.0)

        self.assertEqual(collected.search_result.stats.branches_created, 2)
        self.assertEqual(collected.search_result.stats.branches_faulted, 1)

        faults = [event for event in collected.trace if isinstance(event, BranchFaultTrace)]
        self.assertEqual(len(faults), 1)
        fault = faults[0]
        self.assertEqual(fault.action_id, "strike")
        self.assertEqual(fault.status, "missing_result")
        self.assertIsNone(fault.fault_kind)
        self.assertIsNone(fault.detail)

        ends = [event for event in collected.trace if isinstance(event, SearchTraceEnd)]
        self.assertEqual(len(ends), 1)
        self.assertEqual(ends[0].branches_created, 2)
        self.assertEqual(ends[0].branches_faulted, 1)

        targets = {target.action_id: target for target in collected.targets.root_actions}
        missing = targets["strike"]
        self.assertTrue(missing.evaluated)
        self.assertIsNone(missing.estimated_q)
        self.assertEqual(missing.target_source, "no_target")
        self.assertTrue(missing.censored)
        self.assertEqual(missing.censor_reason, "branch_fault:missing_result")
        self.assertEqual(len(missing.rng_outcomes), 1)
        self.assertIsNone(missing.rng_outcomes[0].value)
        self.assertEqual(missing.rng_outcomes[0].target_source, "no_target")
        self.assertEqual(
            missing.rng_outcomes[0].censor_reason,
            "branch_fault:missing_result",
        )

        clean = targets["end"]
        self.assertTrue(clean.evaluated)
        self.assertIsNotNone(clean.estimated_q)
        self.assertEqual(clean.target_source, "terminal")
        self.assertFalse(clean.censored)

    async def test_all_admitted_results_missing_fails_fast(self) -> None:
        collector = BudgetedOracleCollector(
            _AllMissingBranchResultClient(),
            policy=PriorHeuristicPolicy(),
            value_fn=HeuristicValueFunction(),
            config=OracleCollectionConfig(
                beam_config=BeamSearchConfig(
                    beam_width=2,
                    top_k_actions=2,
                    max_depth=1,
                    release_branches_on_finish=False,
                ),
                target_beam_width=2,
                exhaustive_root_actions=True,
            ),
        )
        root_decision = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": {
                "hp": 40,
                "maxHp": 50,
                "enemies": [{"hp": 10, "maxHp": 10, "isAlive": True, "intent": {}}],
                "legal_actions": _ACTIONS,
            },
        }

        with self.assertRaisesRegex(RuntimeError, "all emulate_actions branch results faulted"):
            await collector.collect("inst-all-missing", root_decision, timeout_s=1.0)


if __name__ == "__main__":
    unittest.main()
