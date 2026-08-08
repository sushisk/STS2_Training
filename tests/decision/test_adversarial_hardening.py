from __future__ import annotations

import unittest
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.api.contract import (
    ApiProtocolError,
    RequestFaultedError,
    RequestRejectedError,
)
from sts2_training.decision.beam_search import BeamSearchConfig, BeamSearchEngine
from sts2_training.decision.engine import CombatDecisionEngine
from sts2_training.decision.policy import ActionCandidate, PolicyModel, PriorHeuristicPolicy
from sts2_training.decision.value import DEFAULT_WEIGHTS, HeuristicValueFunction, ValueModel


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


class _BatchOnlyValue(ValueModel):
    def evaluate(self, masked_emulator_dto: Mapping[str, Any]) -> float:
        raise AssertionError("root should not be scored individually")

    def evaluate_batch(self, dtos: Sequence[Mapping[str, Any]]) -> list[float]:
        return [0.0] * len(dtos)


class _NonFiniteValue(ValueModel):
    def evaluate(self, masked_emulator_dto: Mapping[str, Any]) -> float:
        return 0.0

    def evaluate_batch(self, dtos: Sequence[Mapping[str, Any]]) -> list[float]:
        return [float("nan")] * len(dtos)


class _ExplodingPolicy(PolicyModel):
    def propose(
        self,
        legal_actions: Sequence[Mapping[str, Any]],
        masked_emulator_dto: Mapping[str, Any],
        *,
        top_k: int,
    ) -> list[ActionCandidate]:
        raise RuntimeError("policy bug")


class _FalseyExplodingPolicy(_ExplodingPolicy):
    def __bool__(self) -> bool:
        return False


class _IllegalActionPolicy(PolicyModel):
    def propose(
        self,
        legal_actions: Sequence[Mapping[str, Any]],
        masked_emulator_dto: Mapping[str, Any],
        *,
        top_k: int,
    ) -> list[ActionCandidate]:
        return [ActionCandidate("not-legal")]


class _ProtocolFailureClient(_BeamClient):
    async def emulate_actions(
        self,
        instance_id: str,
        items: Sequence[Mapping[str, Any]],
        *,
        timeout_s: float,
        simulation_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise ApiProtocolError("bad emulate_actions response")


class _FaultedFailureClient(_BeamClient):
    async def emulate_actions(
        self,
        instance_id: str,
        items: Sequence[Mapping[str, Any]],
        *,
        timeout_s: float,
        simulation_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise RequestFaultedError(
            {
                "operation": "emulate_actions",
                "status": "faulted",
                "error": "server fault",
            }
        )


class _InvalidatingRejectedClient(_BeamClient):
    async def emulate_actions(
        self,
        instance_id: str,
        items: Sequence[Mapping[str, Any]],
        *,
        timeout_s: float,
        simulation_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.session_invalid = True
        raise RequestRejectedError(
            {
                "operation": "emulate_actions",
                "status": "rejected",
                "error": "session invalid",
                "fault_kind": "session_instance_conflict",
            }
        )


class _RejectSecondChunkClient(_BeamClient):
    def __init__(self) -> None:
        super().__init__()
        self._calls = 0

    async def emulate_actions(
        self,
        instance_id: str,
        items: Sequence[Mapping[str, Any]],
        *,
        timeout_s: float,
        simulation_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._calls += 1
        if self._calls == 2:
            raise RequestRejectedError(
                {
                    "operation": "emulate_actions",
                    "status": "rejected",
                    "error": "capacity changed",
                    "fault_kind": "branch_capacity",
                }
            )
        return await super().emulate_actions(
            instance_id,
            items,
            timeout_s=timeout_s,
            simulation_options=simulation_options,
        )


class _CleanupBugClient(_BeamClient):
    async def cancel_branches(
        self, instance_id: str, branch_ids: Sequence[str], *, timeout_s: float
    ) -> dict[str, Any]:
        raise RuntimeError("cleanup bug")


class _DecisionClient(_BeamClient):
    def __init__(self) -> None:
        super().__init__()
        self.root_actions: list[dict[str, Any]] = [
            {"action_id": "strike", "action_type": "card", "is_available": True},
            {"action_id": "end", "action_type": "system", "is_available": True},
        ]

    async def get_decision(
        self, instance_id: str, branch_id: str = "root", *, timeout_s: float
    ) -> dict[str, Any]:
        return {
            "decision_point_id": "d-root",
            "masked_emulator_dto": {"legal_actions": self.root_actions},
        }


def _root(actions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "decision_point_id": "d-root",
        "masked_emulator_dto": {"legal_actions": actions},
    }


def _card(action_id: str) -> dict[str, Any]:
    return {"action_id": action_id, "action_type": "card", "is_available": True}


class BeamSearchHardeningTest(unittest.IsolatedAsyncioTestCase):
    async def test_server_batch_cap_overrides_local_default(self) -> None:
        client = _BeamClient(max_items=2)
        actions = [_card("c1"), _card("c2"), _card("c3")]
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
            _root([_card("strike")]),
            timeout_s=1.0,
        )

        self.assertEqual(client.emulate_call_sizes, [1])
        self.assertEqual(result.best_root_action_id, "strike")
        self.assertEqual(result.reason, "not_beam_searchable")

    async def test_unavailable_non_combat_action_does_not_disable_search(self) -> None:
        client = _BeamClient()
        actions = [
            _card("strike"),
            {
                "action_id": "reward-choice",
                "action_type": "choice_reward_card",
                "is_available": False,
            },
        ]
        engine = BeamSearchEngine(
            client,
            policy=PriorHeuristicPolicy(),
            value_fn=HeuristicValueFunction(),
            config=BeamSearchConfig(max_depth=1, top_k_actions=1),
        )

        result = await engine.search("inst-001", _root(actions), timeout_s=1.0)

        self.assertEqual(result.best_root_action_id, "strike")
        self.assertEqual(client.emulate_call_sizes, [1])

    async def test_root_is_not_scored_with_singleton_value_call(self) -> None:
        client = _BeamClient()
        engine = BeamSearchEngine(
            client,
            policy=PriorHeuristicPolicy(),
            value_fn=_BatchOnlyValue(),
            config=BeamSearchConfig(max_depth=1, top_k_actions=1),
        )

        result = await engine.search(
            "inst-001", _root([_card("strike")]), timeout_s=1.0
        )

        self.assertEqual(result.best_root_action_id, "strike")

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
                _root([_card("strike")]),
                timeout_s=1.0,
            )

        self.assertEqual(len(client.cancel_calls), 1)
        self.assertEqual(len(client.release_calls), 1)

    async def test_non_finite_value_fails_loudly(self) -> None:
        client = _BeamClient()
        engine = BeamSearchEngine(
            client,
            policy=PriorHeuristicPolicy(),
            value_fn=_NonFiniteValue(),
            config=BeamSearchConfig(max_depth=1, top_k_actions=1),
        )

        with self.assertRaisesRegex(RuntimeError, "finite numeric"):
            await engine.search(
                "inst-001", _root([_card("strike")]), timeout_s=1.0
            )

    async def test_illegal_policy_action_fails_before_emulation(self) -> None:
        client = _BeamClient()
        engine = BeamSearchEngine(
            client,
            policy=_IllegalActionPolicy(),
            value_fn=HeuristicValueFunction(),
            config=BeamSearchConfig(max_depth=1, top_k_actions=1),
        )

        with self.assertRaisesRegex(RuntimeError, "not currently available"):
            await engine.search(
                "inst-001", _root([_card("strike")]), timeout_s=1.0
            )

        self.assertEqual(client.emulate_call_sizes, [])

    async def test_protocol_error_is_not_converted_to_fallback(self) -> None:
        engine = BeamSearchEngine(
            _ProtocolFailureClient(),
            policy=PriorHeuristicPolicy(),
            value_fn=HeuristicValueFunction(),
            config=BeamSearchConfig(max_depth=1, top_k_actions=1),
        )

        with self.assertRaisesRegex(ApiProtocolError, "bad emulate_actions"):
            await engine.search(
                "inst-001", _root([_card("strike")]), timeout_s=1.0
            )

    async def test_faulted_batch_is_not_converted_to_fallback(self) -> None:
        engine = BeamSearchEngine(
            _FaultedFailureClient(),
            policy=PriorHeuristicPolicy(),
            value_fn=HeuristicValueFunction(),
            config=BeamSearchConfig(max_depth=1, top_k_actions=1),
        )

        with self.assertRaises(RequestFaultedError):
            await engine.search(
                "inst-001", _root([_card("strike")]), timeout_s=1.0
            )

    async def test_session_invalidating_rejection_is_not_converted_to_fallback(self) -> None:
        engine = BeamSearchEngine(
            _InvalidatingRejectedClient(),
            policy=PriorHeuristicPolicy(),
            value_fn=HeuristicValueFunction(),
            config=BeamSearchConfig(max_depth=1, top_k_actions=1),
        )

        with self.assertRaises(RequestRejectedError):
            await engine.search(
                "inst-001", _root([_card("strike")]), timeout_s=1.0
            )

    async def test_partial_depth_rejection_does_not_increment_depths_completed(self) -> None:
        client = _RejectSecondChunkClient()
        engine = BeamSearchEngine(
            client,
            policy=PriorHeuristicPolicy(),
            value_fn=HeuristicValueFunction(),
            config=BeamSearchConfig(max_depth=1, top_k_actions=3, max_batch_size=2),
        )

        result = await engine.search(
            "inst-001",
            _root([_card("c1"), _card("c2"), _card("c3")]),
            timeout_s=1.0,
        )

        self.assertTrue(result.reason.startswith("emulate_actions_rejected"))
        self.assertEqual(result.stats.depths_completed, 0)
        self.assertIsNotNone(result.best_root_action_id)

    async def test_unexpected_cleanup_bug_is_not_swallowed(self) -> None:
        client = _CleanupBugClient()
        engine = BeamSearchEngine(
            client,
            policy=PriorHeuristicPolicy(),
            value_fn=HeuristicValueFunction(),
            config=BeamSearchConfig(max_depth=1, top_k_actions=1),
        )

        with self.assertRaisesRegex(RuntimeError, "cleanup bug"):
            await engine.search(
                "inst-001", _root([_card("strike")]), timeout_s=1.0
            )

    def test_invalid_config_numbers_are_rejected(self) -> None:
        for kwargs in (
            {"beam_width": True},
            {"top_k_actions": 1.5},
            {"max_depth": False},
            {"max_batch_size": 2.5},
            {"time_budget_ms": float("nan")},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    BeamSearchConfig(**kwargs)

    def test_non_positive_time_budget_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BeamSearchConfig(time_budget_ms=0)


class CombatDecisionEngineHardeningTest(unittest.IsolatedAsyncioTestCase):
    async def test_policy_bug_is_not_hidden_by_heuristic_fallback(self) -> None:
        client = _DecisionClient()
        engine = CombatDecisionEngine(client, policy=_ExplodingPolicy())

        with self.assertRaisesRegex(RuntimeError, "policy bug"):
            await engine.decide("inst-001", timeout_s=1.0)

    async def test_falsey_custom_policy_is_not_replaced_by_default(self) -> None:
        client = _DecisionClient()
        engine = CombatDecisionEngine(client, policy=_FalseyExplodingPolicy())

        with self.assertRaisesRegex(RuntimeError, "policy bug"):
            await engine.decide("inst-001", timeout_s=1.0)

    async def test_single_available_action_is_forced(self) -> None:
        client = _DecisionClient()
        client.root_actions = [
            {
                "action_id": "unavailable",
                "action_type": "choice_reward_card",
                "is_available": False,
            },
            {"action_id": "end", "action_type": "system", "is_available": True},
        ]
        engine = CombatDecisionEngine(client)

        outcome = await engine.decide("inst-001", timeout_s=1.0)

        self.assertEqual(outcome.source, "forced_single_action")
        self.assertEqual(outcome.chosen_action_id, "end")
        self.assertEqual(client.emulate_call_sizes, [])


class HeuristicValueHardeningTest(unittest.TestCase):
    def test_non_combat_transition_false_is_not_treated_as_defeat(self) -> None:
        value_fn = HeuristicValueFunction()
        dto = {
            "transition": {"kind": "room_changed", "victory": False},
            "hp": 50,
            "maxHp": 50,
            "enemies": [],
        }

        self.assertNotEqual(value_fn.evaluate(dto), DEFAULT_WEIGHTS["defeat_penalty"])

    def test_killing_enemy_cannot_make_enemy_hp_progress_worse(self) -> None:
        value_fn = HeuristicValueFunction()
        before = {
            "hp": 50,
            "maxHp": 50,
            "enemies": [
                {"hp": 1, "maxHp": 100, "isAlive": True},
                {"hp": 100, "maxHp": 100, "isAlive": True},
            ],
        }
        after = {
            "hp": 50,
            "maxHp": 50,
            "enemies": [
                {"hp": 0, "maxHp": 100, "isAlive": False},
                {"hp": 100, "maxHp": 100, "isAlive": True},
            ],
        }

        self.assertGreater(value_fn.evaluate(after), value_fn.evaluate(before))


if __name__ == "__main__":
    unittest.main()
