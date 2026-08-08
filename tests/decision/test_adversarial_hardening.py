from __future__ import annotations

import unittest
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.decision.beam_search import BeamSearchConfig, BeamSearchEngine
from sts2_training.decision.engine import CombatDecisionEngine
from sts2_training.decision.policy import ActionCandidate, PolicyModel, PriorHeuristicPolicy
from sts2_training.decision.value import HeuristicValueFunction, ValueModel


class _BeamClient:
    pending_retry = None
    session_invalid = False

    def __init__(self, *, max_items: int = 64) -> None:
        self.max_emulate_actions_items = max_items
        self.emulate_call_sizes: list[int] = []
        self.cancel_calls: list[list[str]] = []
        self.release_calls: list[list[str]] = []
        self.child_dto: Mapping[str, Any] = {
            "hp": 40,
            "maxHp": 50,
            "enemies": [],
            "legal_actions": [],
        }

    async def emulate_actions(
        self,
        instance_id: str,
        items: Sequence[Mapping[str, Any]],
        *,
        timeout_s: float,
        simulation_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.emulate_call_sizes.append(len(items))
        return {
            "branch_results": {
                item["branch_id"]: {
                    "status": "completed",
                    "decision_point_id": f"next-{item['branch_id']}",
                    "masked_emulator_dto": dict(self.child_dto),
                    "branch_log": [],
                }
                for item in items
            }
        }

    async def cancel_branches(
        self, instance_id: str, branch_ids: Sequence[str], *, timeout_s: float
    ) -> dict[str, Any]:
        self.cancel_calls.append(list(branch_ids))
        return {}

    async def release_branches(
        self, instance_id: str, branch_ids: Sequence[str], *, timeout_s: float
    ) -> dict[str, Any]:
        self.release_calls.append(list(branch_ids))
        return {}


class _ShortBatchValue(ValueModel):
    def evaluate(self, masked_emulator_dto: Mapping[str, Any]) -> float:
        return 0.0

    def evaluate_batch(self, dtos: Sequence[Mapping[str, Any]]) -> list[float]:
        return []


class _ExplodingPolicy(PolicyModel):
    def propose(
        self,
        legal_actions: Sequence[Mapping[str, Any]],
        masked_emulator_dto: Mapping[str, Any],
        *,
        top_k: int,
    ) -> list[ActionCandidate]:
        raise RuntimeError("policy bug")


class _DecisionClient(_BeamClient):
    async def get_decision(
        self, instance_id: str, branch_id: str = "root", *, timeout_s: float
    ) -> dict[str, Any]:
        return {
            "decision_point_id": "d-root",
            "masked_emulator_dto": {
                "legal_actions": [
                    {"action_id": "strike", "action_type": "card", "is_available": True},
                    {"action_id": "end", "action_type": "system", "is_available": True},
                ]
            },
        }


def _root(actions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "decision_point_id": "d-root",
        "masked_emulator_dto": {"legal_actions": actions},
    }


class BeamSearchHardeningTest(unittest.IsolatedAsyncioTestCase):
    async def test_server_batch_cap_overrides_local_default(self) -> None:
        client = _BeamClient(max_items=2)
        actions = [
            {"action_id": "c1", "action_type": "card", "is_available": True},
            {"action_id": "c2", "action_type": "card", "is_available": True},
            {"action_id": "c3", "action_type": "card", "is_available": True},
        ]
        engine = BeamSearchEngine(
            client,
            policy=PriorHeuristicPolicy(),
            value_fn=HeuristicValueFunction(),
            config=BeamSearchConfig(max_depth=1, top_k_actions=3, max_batch_size=64),
        )

        await engine.search("inst-001", _root(actions), timeout_s=1.0)

        self.assertEqual(client.emulate_call_sizes, [2, 1])

    async def test_non_combat_boundary_is_not_expanded_at_deeper_depth(self) -> None:
        client = _BeamClient()
        client.child_dto = {
            "hp": 40,
            "maxHp": 50,
            "enemies": [],
            "legal_actions": [
                {
                    "action_id": "reward-choice",
                    "action_type": "choice_reward_card",
                    "is_available": True,
                }
            ],
        }
        engine = BeamSearchEngine(
            client,
            policy=PriorHeuristicPolicy(),
            value_fn=HeuristicValueFunction(),
            config=BeamSearchConfig(max_depth=2, top_k_actions=1),
        )

        result = await engine.search(
            "inst-001",
            _root([{"action_id": "strike", "action_type": "card", "is_available": True}]),
            timeout_s=1.0,
        )

        self.assertEqual(client.emulate_call_sizes, [1])
        self.assertEqual(result.best_root_action_id, "strike")
        self.assertEqual(result.reason, "not_beam_searchable")

    async def test_value_batch_cardinality_mismatch_fails_loudly(self) -> None:
        client = _BeamClient()
        engine = BeamSearchEngine(
            client,
            policy=PriorHeuristicPolicy(),
            value_fn=_ShortBatchValue(),
            config=BeamSearchConfig(max_depth=1, top_k_actions=1),
        )

        with self.assertRaisesRegex(RuntimeError, "exactly one value"):
            await engine.search(
                "inst-001",
                _root([{"action_id": "strike", "action_type": "card", "is_available": True}]),
                timeout_s=1.0,
            )

        self.assertEqual(len(client.cancel_calls), 1)
        self.assertEqual(len(client.release_calls), 1)

    def test_non_positive_time_budget_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BeamSearchConfig(time_budget_ms=0)


class CombatDecisionEngineHardeningTest(unittest.IsolatedAsyncioTestCase):
    async def test_policy_bug_is_not_hidden_by_heuristic_fallback(self) -> None:
        client = _DecisionClient()
        engine = CombatDecisionEngine(client, policy=_ExplodingPolicy())

        with self.assertRaisesRegex(RuntimeError, "policy bug"):
            await engine.decide("inst-001", timeout_s=1.0)


if __name__ == "__main__":
    unittest.main()
