from __future__ import annotations

import unittest
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.decision.beam_search import (
    BeamSearchConfig,
    BeamSearchEngine,
    BeamSearchResult,
    BeamSearchStats,
)
from sts2_training.decision.engine import CombatDecisionEngine
from sts2_training.decision.policy import ActionCandidate, PolicyModel
from sts2_training.decision.value import HeuristicValueFunction, ValueModel


class _BeamClient:
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
        return {}

    async def release_branches(
        self,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        return {}


class _CleanupBugClient(_BeamClient):
    async def cancel_branches(
        self,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        raise RuntimeError("cleanup bug")


class _BatchOnlyPolicy(PolicyModel):
    def propose_batch(
        self,
        requests: Sequence[tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]]],
        *,
        top_k: int,
    ) -> list[list[ActionCandidate]]:
        return [
            [ActionCandidate(action_id=legal_actions[0]["action_id"])]
            for legal_actions, _dto in requests
        ]


class _BatchOnlyValue(ValueModel):
    def evaluate_batch(self, dtos: Sequence[Mapping[str, Any]]) -> list[float]:
        return [1.0] * len(dtos)


class _DecisionClient:
    async def get_decision(
        self,
        instance_id: str,
        branch_id: str = "root",
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        return {
            "decision_point_id": "d-root",
            "masked_emulator_dto": {
                "legal_actions": [
                    {"action_id": "beam-a", "action_type": "card", "is_available": True},
                    {"action_id": "fallback-b", "action_type": "system", "is_available": True},
                ]
            },
        }


class _RejectedPartialBeam:
    async def search(
        self,
        instance_id: str,
        root_decision: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> BeamSearchResult:
        return BeamSearchResult(
            best_root_action_id="beam-a",
            best_value=10.0,
            best_node=None,
            reason="emulate_actions_rejected:branch_capacity",
            stats=BeamSearchStats(depths_completed=0),
        )


class _FallbackSelector:
    def select(self, legal_actions: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        return legal_actions[1]


def _root() -> dict[str, Any]:
    return {
        "decision_point_id": "d-root",
        "masked_emulator_dto": {
            "legal_actions": [
                {"action_id": "strike", "action_type": "card", "is_available": True}
            ]
        },
    }


class CleanupExceptionBoundaryTest(unittest.IsolatedAsyncioTestCase):
    async def test_outer_exception_context_does_not_hide_cleanup_failure(self) -> None:
        engine = BeamSearchEngine(
            _CleanupBugClient(),
            policy=_BatchOnlyPolicy(),
            value_fn=_BatchOnlyValue(),
            config=BeamSearchConfig(max_depth=1, top_k_actions=1),
        )

        try:
            raise ValueError("unrelated caller exception")
        except ValueError:
            with self.assertRaisesRegex(RuntimeError, "cleanup bug"):
                await engine.search("inst-001", _root(), timeout_s=1.0)


class BatchOnlyModelContractTest(unittest.IsolatedAsyncioTestCase):
    async def test_batch_only_policy_and_value_do_not_require_dummy_scalar_methods(self) -> None:
        engine = BeamSearchEngine(
            _BeamClient(),
            policy=_BatchOnlyPolicy(),
            value_fn=_BatchOnlyValue(),
            config=BeamSearchConfig(max_depth=1, top_k_actions=1),
        )

        result = await engine.search("inst-001", _root(), timeout_s=1.0)

        self.assertEqual(result.best_root_action_id, "strike")


class IncompleteBeamDecisionTest(unittest.IsolatedAsyncioTestCase):
    async def test_rejected_partial_depth_uses_heuristic_fallback(self) -> None:
        engine = CombatDecisionEngine(
            _DecisionClient(),
            fallback_selector=_FallbackSelector(),  # type: ignore[arg-type]
        )
        engine._beam = _RejectedPartialBeam()  # type: ignore[assignment]

        outcome = await engine.decide("inst-001", timeout_s=1.0)

        self.assertEqual(outcome.source, "heuristic_fallback")
        self.assertEqual(outcome.chosen_action_id, "fallback-b")
        self.assertIsNotNone(outcome.beam_result)


class HeuristicInputValidationTest(unittest.TestCase):
    def test_non_numeric_present_field_is_not_silently_defaulted(self) -> None:
        value_fn = HeuristicValueFunction()

        with self.assertRaisesRegex(ValueError, "finite numeric"):
            value_fn.evaluate({"hp": "40", "maxHp": 50, "enemies": []})

    def test_huge_integer_reports_contract_error_not_overflow_error(self) -> None:
        value_fn = HeuristicValueFunction()

        with self.assertRaisesRegex(ValueError, "finite numeric"):
            value_fn.evaluate({"hp": 10**10000, "maxHp": 50, "enemies": []})


if __name__ == "__main__":
    unittest.main()
