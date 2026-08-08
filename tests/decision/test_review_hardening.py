from __future__ import annotations

import unittest
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.api.contract import RequestRejectedError
from sts2_training.decision.beam_search import BeamSearchConfig, BeamSearchEngine
from sts2_training.decision.engine import CombatDecisionEngine
from sts2_training.decision.policy import PriorHeuristicPolicy
from sts2_training.decision.value import HeuristicValueFunction


class _CleanupRejectClient:
    pending_retry = None
    session_invalid = False
    max_emulate_actions_items = 64

    def __init__(self) -> None:
        self.release_calls: list[list[str]] = []

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
                    "status": "completed",
                    "decision_point_id": f"next-{item['branch_id']}",
                    "masked_emulator_dto": {
                        "hp": 40,
                        "maxHp": 50,
                        "enemies": [],
                        "legal_actions": [],
                    },
                    "branch_log": [],
                }
                for item in items
            }
        }

    async def cancel_branches(
        self,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        raise RequestRejectedError(
            {
                "operation": "cancel_branches",
                "status": "rejected",
                "error": "cleanup rejected",
                "fault_kind": "branch_capacity",
            }
        )

    async def release_branches(
        self,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        self.release_calls.append(list(branch_ids))
        return {}


class _DecisionClient:
    pending_retry = None
    session_invalid = False

    def __init__(self) -> None:
        self.get_calls = 0

    async def get_decision(
        self,
        instance_id: str,
        branch_id: str = "root",
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        self.get_calls += 1
        return {
            "decision_point_id": "d-root",
            "masked_emulator_dto": {
                "legal_actions": [
                    {
                        "action_id": "reward-a",
                        "action_type": "choice_reward_card",
                        "is_available": True,
                    },
                    {
                        "action_id": "reward-b",
                        "action_type": "choice_reward_card",
                        "is_available": True,
                    },
                ]
            },
        }


class _NonMappingDecisionClient(_DecisionClient):
    async def get_decision(
        self,
        instance_id: str,
        branch_id: str = "root",
        *,
        timeout_s: float,
    ) -> Any:
        self.get_calls += 1
        return None


class _MissingDecisionPointClient(_DecisionClient):
    async def get_decision(
        self,
        instance_id: str,
        branch_id: str = "root",
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        decision = await super().get_decision(
            instance_id, branch_id, timeout_s=timeout_s
        )
        decision.pop("decision_point_id")
        return decision


class _MissingLegalActionsClient(_DecisionClient):
    async def get_decision(
        self,
        instance_id: str,
        branch_id: str = "root",
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        decision = await super().get_decision(
            instance_id, branch_id, timeout_s=timeout_s
        )
        decision["masked_emulator_dto"] = {}
        return decision


class _MalformedLegalActionClient(_DecisionClient):
    async def get_decision(
        self,
        instance_id: str,
        branch_id: str = "root",
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        decision = await super().get_decision(
            instance_id, branch_id, timeout_s=timeout_s
        )
        decision["masked_emulator_dto"]["legal_actions"] = [
            {
                "action_id": "reward-a",
                "action_type": "choice_reward_card",
                "is_available": True,
            },
            42,
        ]
        return decision


class _InvalidFallbackSelector:
    def select(self, legal_actions: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        return {"action_id": "not-legal"}


def _root() -> dict[str, Any]:
    return {
        "decision_point_id": "d-root",
        "masked_emulator_dto": {
            "legal_actions": [
                {"action_id": "strike", "action_type": "card", "is_available": True}
            ]
        },
    }


class BeamCleanupReviewTest(unittest.IsolatedAsyncioTestCase):
    async def test_release_is_not_attempted_when_cancellation_did_not_complete(self) -> None:
        client = _CleanupRejectClient()
        engine = BeamSearchEngine(
            client,
            policy=PriorHeuristicPolicy(),
            value_fn=HeuristicValueFunction(),
            config=BeamSearchConfig(max_depth=1, top_k_actions=1),
        )

        result = await engine.search("inst-001", _root(), timeout_s=1.0)

        self.assertEqual(result.best_root_action_id, "strike")
        self.assertEqual(client.release_calls, [])


class CombatDecisionReviewTest(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_fallback_action_is_rejected_before_commit(self) -> None:
        client = _DecisionClient()
        engine = CombatDecisionEngine(
            client,
            fallback_selector=_InvalidFallbackSelector(),  # type: ignore[arg-type]
        )

        with self.assertRaisesRegex(RuntimeError, "available legal action"):
            await engine.decide("inst-001", timeout_s=1.0)

    async def test_non_mapping_decision_is_rejected_at_boundary(self) -> None:
        engine = CombatDecisionEngine(_NonMappingDecisionClient())

        with self.assertRaisesRegex(RuntimeError, "must return a mapping"):
            await engine.decide("inst-001", timeout_s=1.0)

    async def test_missing_decision_point_id_is_rejected_at_boundary(self) -> None:
        engine = CombatDecisionEngine(_MissingDecisionPointClient())

        with self.assertRaisesRegex(RuntimeError, "decision_point_id"):
            await engine.decide("inst-001", timeout_s=1.0)

    async def test_missing_legal_actions_is_rejected_for_nonterminal_decision(self) -> None:
        engine = CombatDecisionEngine(_MissingLegalActionsClient())

        with self.assertRaisesRegex(RuntimeError, "invalid legal_actions"):
            await engine.decide("inst-001", timeout_s=1.0)

    async def test_malformed_legal_action_is_rejected_at_boundary(self) -> None:
        engine = CombatDecisionEngine(_MalformedLegalActionClient())

        with self.assertRaisesRegex(RuntimeError, r"legal_actions\[1\]"):
            await engine.decide("inst-001", timeout_s=1.0)

    async def test_non_finite_and_boolean_timeouts_fail_before_api_call(self) -> None:
        client = _DecisionClient()
        engine = CombatDecisionEngine(client)

        for timeout_s in (float("nan"), float("inf"), True):
            with self.subTest(timeout_s=timeout_s):
                with self.assertRaisesRegex(ValueError, "finite positive"):
                    await engine.decide("inst-001", timeout_s=timeout_s)

        self.assertEqual(client.get_calls, 0)


class HeuristicValueReviewTest(unittest.TestCase):
    def test_unknown_weight_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown heuristic weight"):
            HeuristicValueFunction(weights={"player_hp_rato": 1.0})

    def test_non_finite_weight_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite number"):
            HeuristicValueFunction(weights={"player_hp_ratio": float("nan")})

    def test_non_finite_dto_number_is_rejected(self) -> None:
        value_fn = HeuristicValueFunction()

        with self.assertRaisesRegex(ValueError, "input numbers must be finite"):
            value_fn.evaluate({"hp": float("inf"), "maxHp": 50, "enemies": []})


if __name__ == "__main__":
    unittest.main()
