from __future__ import annotations

import unittest
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.decision.beam_search import BeamSearchConfig, BeamSearchEngine
from sts2_training.decision.policy import PriorHeuristicPolicy
from sts2_training.decision.value import HeuristicValueFunction


class _AllFaultedClient:
    pending_retry = None
    session_invalid = False
    max_emulate_actions_items = 64

    async def emulate_actions(
        self,
        instance_id: str,
        items: Sequence[Mapping[str, Any]],
        *,
        timeout_s: float,
        simulation_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "branch_results": {
                item["branch_id"]: {
                    "status": "faulted",
                    "error": "emulator failure",
                }
                for item in items
            }
        }

    async def cancel_branches(
        self, instance_id: str, branch_ids: Sequence[str], *, timeout_s: float
    ) -> dict[str, Any]:
        return {}

    async def release_branches(
        self, instance_id: str, branch_ids: Sequence[str], *, timeout_s: float
    ) -> dict[str, Any]:
        return {}


class FaultBoundaryTest(unittest.IsolatedAsyncioTestCase):
    async def test_all_faulted_branches_are_not_converted_to_heuristic_fallback(self) -> None:
        engine = BeamSearchEngine(
            _AllFaultedClient(),
            policy=PriorHeuristicPolicy(),
            value_fn=HeuristicValueFunction(),
            config=BeamSearchConfig(max_depth=1, top_k_actions=1),
        )
        root_decision = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": {
                "legal_actions": [
                    {"action_id": "strike", "action_type": "card", "is_available": True}
                ]
            },
        }

        with self.assertRaisesRegex(RuntimeError, "all emulate_actions branch results faulted"):
            await engine.search("inst-001", root_decision, timeout_s=1.0)


if __name__ == "__main__":
    unittest.main()
