from __future__ import annotations

import unittest

from sts2_training.decision.beam_search import BeamSearchConfig, BeamSearchEngine
from sts2_training.decision.policy import PriorHeuristicPolicy
from sts2_training.decision.search_trace import InMemorySearchTraceCollector, SearchTraceEnd
from sts2_training.decision.value import HeuristicValueFunction


_ACTIONS = [
    {"action_id": "strike", "action_type": "card", "is_available": True},
    {"action_id": "end", "action_type": "system", "is_available": True},
]


class _Client:
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
        branch_results: dict[str, dict] = {}
        for item in items:
            if item["action_id"] == "strike":
                branch_results[item["branch_id"]] = {
                    "status": "faulted",
                    "fault_kind": "action_fault",
                    "error": "boom",
                }
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


class SearchTraceEndFaultAggregateTest(unittest.IsolatedAsyncioTestCase):
    async def test_search_end_reports_faulted_branch_aggregate(self) -> None:
        collector = InMemorySearchTraceCollector()
        engine = BeamSearchEngine(
            _Client(),
            policy=PriorHeuristicPolicy(),
            value_fn=HeuristicValueFunction(),
            config=BeamSearchConfig(
                max_depth=1,
                top_k_actions=2,
                release_branches_on_finish=False,
            ),
            trace_collector=collector,
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

        result = await engine.search("inst-001", root_decision, timeout_s=1.0)

        # end resolves once; strike consumes all three physical attempts.
        self.assertEqual(result.stats.branches_created, 4)
        self.assertEqual(result.stats.branches_faulted, 1)
        self.assertEqual(result.stats.branch_retry_faults, 2)
        self.assertEqual(result.stats.branch_retry_recoveries, 0)
        end_events = [event for event in collector.events if isinstance(event, SearchTraceEnd)]
        self.assertEqual(len(end_events), 1)
        self.assertEqual(end_events[0].branches_created, 4)
        self.assertEqual(end_events[0].branches_faulted, 1)


if __name__ == "__main__":
    unittest.main()
