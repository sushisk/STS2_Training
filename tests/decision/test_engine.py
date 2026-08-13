"""Coverage for `CombatDecisionEngine`'s get_decision -> (beam search |
heuristic fallback) -> commit_action wiring, against the same fake RL
connection style as `test_beam_search.py`.
"""

from __future__ import annotations

import random
import unittest

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.api.contract import MASK_VERSION, SCHEMA_VERSION
from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.decision.combat_decision import COMBAT_BEAM_ACTION_TYPES
from sts2_training.decision.engine import CombatDecisionEngine
from sts2_training.decision.policy import PriorHeuristicPolicy
from sts2_training.decision.value import HeuristicValueFunction
from sts2_training.selection.heuristic_selector import HeuristicCombatSelector


def _common(request: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "server_epoch": "epoch-1",
        "client_session_id": request["client_session_id"],
        "request_seq": request["request_seq"],
        "request_id": request["request_id"],
        "operation": request["operation"],
    }


def _wire_decision(decision: dict) -> dict:
    result = dict(decision)
    dto = result.get("masked_emulator_dto")
    if isinstance(dto, dict):
        result["masked_emulator_dto"] = {"mask_version": MASK_VERSION, **dto}
    return result


class _FakeConnection:
    client_session_id = "session-a"

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.root_decision: dict = {}
        self.emulate_results: dict[tuple[str, str], dict] = {}
        self.reject_emulate_actions = False
        self.committed_action_ids: list[str] = []

    async def exchange(self, message: dict, *, deadline: float) -> dict:
        request = dict(message)
        self.messages.append(request)
        operation = request["operation"]

        if operation == "start_instance":
            return {
                **_common(request),
                "status": "completed",
                "instance_id": "inst-001",
                "max_emulate_actions_items": 64,
            }

        if operation == "get_decision":
            return {
                **_common(request),
                "instance_id": request["instance_id"],
                "status": "completed",
                "branch_id": "root",
                **_wire_decision(self.root_decision),
            }

        if operation == "commit_action":
            self.committed_action_ids.append(request["action_id"])
            return {
                **_common(request),
                "instance_id": request["instance_id"],
                "status": "completed",
                "branch_id": "root",
                "decision_point_id": "d-root-2",
                "branch_log": [],
                "masked_emulator_dto": {
                    "mask_version": MASK_VERSION,
                    "legal_actions": [
                        {"action_id": "noop", "action_type": "system"}
                    ],
                },
            }

        if operation == "emulate_actions":
            if self.reject_emulate_actions:
                return {
                    **_common(request),
                    "instance_id": request["instance_id"],
                    "status": "rejected",
                    "error": "boom",
                    "fault_kind": "stale_decision_point",
                }
            branch_results = {}
            for item in request["items"]:
                key = (item["parent_branch_id"], item["action_id"])
                canned = self.emulate_results[key]
                branch_results[item["branch_id"]] = {
                    "branch_id": item["branch_id"],
                    "parent_branch_id": item["parent_branch_id"],
                    "rng_id": item["rng_id"],
                    "branch_log": [],
                    **canned,
                }
            return {
                **_common(request),
                "instance_id": request["instance_id"],
                "status": "completed",
                "branch_results": branch_results,
            }

        if operation in ("cancel_branches", "release_branches"):
            statuses = {
                bid: ("cancelled" if operation == "cancel_branches" else "released")
                for bid in request["branch_ids"]
            }
            return {
                **_common(request),
                "instance_id": request["instance_id"],
                "status": "completed",
                "branch_statuses": statuses,
            }

        raise AssertionError(f"unexpected operation: {operation}")

    async def invalidate(self) -> None:
        pass

    async def close(self) -> None:
        pass


def _victory_dto() -> dict:
    return {
        "mask_version": MASK_VERSION,
        "terminal": True,
        "outcome": "victory",
        "hp": 40,
        "maxHp": 50,
        "enemies": [],
        "legal_actions": [],
    }


def _defeat_dto() -> dict:
    return {
        "mask_version": MASK_VERSION,
        "terminal": True,
        "outcome": "defeat",
        "hp": 0,
        "maxHp": 50,
        "enemies": [{"hp": 10, "maxHp": 10, "isAlive": True, "intent": {}}],
        "legal_actions": [],
    }


def _alive_dto() -> dict:
    return {
        "mask_version": MASK_VERSION,
        "hp": 40,
        "maxHp": 50,
        "enemies": [{"hp": 10, "maxHp": 10, "isAlive": True, "intent": {}}],
        "legal_actions": _COMBAT_ACTIONS,
    }


_COMBAT_ACTIONS = [
    {"action_id": "strike", "action_type": "card", "is_available": True},
    {"action_id": "end", "action_type": "system", "is_available": True},
]

_LEGACY_COMBAT_ACTION_TYPES = frozenset({"system", "card", "potion"})


class CombatDecisionEngineTest(unittest.IsolatedAsyncioTestCase):
    async def _client(self, connection: _FakeConnection) -> AsyncTrainingApiClient:
        client = AsyncTrainingApiClient(connection)  # type: ignore[arg-type]
        await client.start_instance({"instance_type": "combat"}, timeout_s=1.0)
        return client

    async def test_decide_and_commit_uses_beam_search_winner(self) -> None:
        connection = _FakeConnection()
        connection.root_decision = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": {"legal_actions": _COMBAT_ACTIONS},
        }
        connection.emulate_results = {
            ("root", "strike"): {
                "status": "completed",
                "decision_point_id": "d-b1",
                "masked_emulator_dto": _victory_dto(),
            },
            ("root", "end"): {
                "status": "completed",
                "decision_point_id": "d-b2",
                "masked_emulator_dto": _alive_dto(),
            },
        }
        client = await self._client(connection)
        engine = CombatDecisionEngine(
            client,
            policy=PriorHeuristicPolicy(),
            value_fn=HeuristicValueFunction(),
            beam_config=BeamSearchConfig(max_depth=1, top_k_actions=2, beam_width=8),
        )

        outcome = await engine.decide("inst-001", timeout_s=5.0)

        self.assertEqual(outcome.source, "beam_search")
        self.assertEqual(outcome.chosen_action_id, "strike")

        response = await engine.decide_and_commit("inst-001", timeout_s=5.0)
        self.assertEqual(response["status"], "completed")
        self.assertEqual(connection.committed_action_ids, ["strike"])

    async def test_default_engine_beam_searches_choice_target(self) -> None:
        connection = _FakeConnection()
        actions = [
            {
                "action_id": "target-good",
                "action_type": "choice_target",
                "is_available": True,
            },
            {
                "action_id": "target-bad",
                "action_type": "choice_target",
                "is_available": True,
            },
        ]
        connection.root_decision = {
            "decision_point_id": "d-root-target",
            "masked_emulator_dto": {"legal_actions": actions},
        }
        connection.emulate_results = {
            ("root", "target-good"): {
                "status": "completed",
                "decision_point_id": "d-target-good",
                "masked_emulator_dto": _victory_dto(),
            },
            ("root", "target-bad"): {
                "status": "completed",
                "decision_point_id": "d-target-bad",
                "masked_emulator_dto": _defeat_dto(),
            },
        }
        client = await self._client(connection)
        engine = CombatDecisionEngine(client)

        outcome = await engine.decide("inst-001", timeout_s=5.0)

        self.assertEqual(outcome.source, "beam_search")
        self.assertEqual(outcome.chosen_action_id, "target-good")
        self.assertIn("emulate_actions", [m["operation"] for m in connection.messages])

    async def test_default_engine_beam_searches_choice_card(self) -> None:
        connection = _FakeConnection()
        actions = [
            {
                "action_id": "choice-good",
                "action_type": "choice_card",
                "is_available": True,
            },
            {
                "action_id": "choice-bad",
                "action_type": "choice_card",
                "is_available": True,
            },
        ]
        connection.root_decision = {
            "decision_point_id": "d-root-card-choice",
            "masked_emulator_dto": {"legal_actions": actions},
        }
        connection.emulate_results = {
            ("root", "choice-good"): {
                "status": "completed",
                "decision_point_id": "d-choice-good",
                "masked_emulator_dto": _victory_dto(),
            },
            ("root", "choice-bad"): {
                "status": "completed",
                "decision_point_id": "d-choice-bad",
                "masked_emulator_dto": _defeat_dto(),
            },
        }
        client = await self._client(connection)
        engine = CombatDecisionEngine(client)

        outcome = await engine.decide("inst-001", timeout_s=5.0)

        self.assertEqual(outcome.source, "beam_search")
        self.assertEqual(outcome.chosen_action_id, "choice-good")
        self.assertIn("emulate_actions", [m["operation"] for m in connection.messages])

    async def test_explicit_beam_config_scope_is_preserved(self) -> None:
        connection = _FakeConnection()
        actions = [
            {
                "action_id": "target-a",
                "action_type": "choice_target",
                "is_available": True,
            },
            {
                "action_id": "target-b",
                "action_type": "choice_target",
                "is_available": True,
            },
        ]
        connection.root_decision = {
            "decision_point_id": "d-root-target",
            "masked_emulator_dto": {"legal_actions": actions},
        }
        client = await self._client(connection)
        engine = CombatDecisionEngine(
            client,
            beam_config=BeamSearchConfig(
                max_depth=1,
                beam_searchable_action_types=_LEGACY_COMBAT_ACTION_TYPES,
            ),
            fallback_selector=HeuristicCombatSelector(rng=random.Random(0)),
        )

        outcome = await engine.decide("inst-001", timeout_s=5.0)

        self.assertEqual(
            engine.beam_search.config.beam_searchable_action_types,
            _LEGACY_COMBAT_ACTION_TYPES,
        )
        self.assertEqual(outcome.source, "heuristic_fallback")
        self.assertNotIn("emulate_actions", [m["operation"] for m in connection.messages])

    def test_conflicting_explicit_beam_scope_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "beam_action_types conflicts"):
            CombatDecisionEngine(
                object(),
                beam_config=BeamSearchConfig(
                    beam_searchable_action_types=_LEGACY_COMBAT_ACTION_TYPES
                ),
                beam_action_types=COMBAT_BEAM_ACTION_TYPES,
            )

    async def test_falls_back_to_heuristic_when_batch_rejected(self) -> None:
        connection = _FakeConnection()
        connection.root_decision = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": {"legal_actions": _COMBAT_ACTIONS},
        }
        connection.reject_emulate_actions = True
        client = await self._client(connection)
        engine = CombatDecisionEngine(
            client,
            fallback_selector=HeuristicCombatSelector(rng=random.Random(0)),
            beam_config=BeamSearchConfig(max_depth=1, top_k_actions=2, beam_width=8),
        )

        outcome = await engine.decide("inst-001", timeout_s=5.0)

        self.assertEqual(outcome.source, "heuristic_fallback")
        self.assertIn(outcome.chosen_action_id, {"strike", "end"})

    async def test_skips_beam_search_for_non_combat_decision(self) -> None:
        connection = _FakeConnection()
        connection.root_decision = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": {
                "legal_actions": [
                    {
                        "action_id": "opt-1",
                        "action_type": "choice_reward_card",
                        "is_available": True,
                    },
                    {
                        "action_id": "opt-2",
                        "action_type": "choice_reward_card",
                        "is_available": True,
                    },
                ]
            },
        }
        client = await self._client(connection)
        engine = CombatDecisionEngine(
            client,
            fallback_selector=HeuristicCombatSelector(rng=random.Random(0)),
        )

        outcome = await engine.decide("inst-001", timeout_s=5.0)

        self.assertEqual(outcome.source, "heuristic_fallback")
        self.assertEqual(
            [m["operation"] for m in connection.messages],
            ["start_instance", "get_decision"],
        )

    async def test_forced_single_action_skips_beam_search_and_fallback(self) -> None:
        connection = _FakeConnection()
        connection.root_decision = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": {
                "legal_actions": [
                    {"action_id": "only", "action_type": "system", "is_available": True}
                ]
            },
        }
        client = await self._client(connection)
        engine = CombatDecisionEngine(client)

        outcome = await engine.decide("inst-001", timeout_s=5.0)

        self.assertEqual(outcome.source, "forced_single_action")
        self.assertEqual(outcome.chosen_action_id, "only")
        self.assertEqual(
            [m["operation"] for m in connection.messages],
            ["start_instance", "get_decision"],
        )

    async def test_no_legal_actions_yields_none_outcome(self) -> None:
        connection = _FakeConnection()
        connection.root_decision = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": {"run_terminal": True, "outcome": "victory"},
        }
        client = await self._client(connection)
        engine = CombatDecisionEngine(client)

        outcome = await engine.decide("inst-001", timeout_s=5.0)

        self.assertEqual(outcome.source, "none")
        self.assertIsNone(outcome.chosen_action_id)


if __name__ == "__main__":
    unittest.main()
