"""Coverage for `BeamSearchEngine` against a fake RL connection wired through
the real `AsyncTrainingApiClient` (so request_seq/session/response validation
all run exactly as in production - see `tests/api/test_emulate_actions_batch.py`
for the same pattern this borrows).
"""

from __future__ import annotations

import unittest

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.api.contract import MASK_VERSION, SCHEMA_VERSION
from sts2_training.decision.beam_search import BeamNode, BeamSearchConfig, BeamSearchEngine
from sts2_training.decision.policy import ActionCandidate, PriorHeuristicPolicy
from sts2_training.decision.search_trace import BranchFaultTrace, InMemorySearchTraceCollector
from sts2_training.decision.value import HeuristicValueFunction


def _common(request: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "server_epoch": "epoch-1",
        "client_session_id": request["client_session_id"],
        "request_seq": request["request_seq"],
        "request_id": request["request_id"],
        "operation": request["operation"],
    }


class _FakeConnection:
    """Minimal scripted RL stand-in. `emulate_results` maps
    `(parent_branch_id, action_id) -> branch_result fields` (status/dto/etc,
    everything except branch_id/parent_branch_id/rng_id/branch_log, which this
    fake fills in from the request itself).
    """

    client_session_id = "session-a"

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.decisions: dict[str, dict] = {}
        self.emulate_results: dict[tuple[str, str], dict] = {}
        # None models legacy Whole Run servers that do not advertise this capability.
        self.max_emulate_actions_items: int | None = 64
        self.reject_emulate_actions = False
        # 1-indexed emulate_actions call number at which rejection starts (for testing
        # a chunk failing partway through a depth's batch); None = never reject.
        self.reject_emulate_actions_from_call: int | None = None
        self._emulate_actions_call_count = 0
        self.cancel_calls: list[list[str]] = []
        self.release_calls: list[list[str]] = []

    async def exchange(self, message: dict, *, deadline: float) -> dict:
        request = dict(message)
        self.messages.append(request)
        operation = request["operation"]

        if operation == "start_instance":
            response = {
                **_common(request),
                "status": "completed",
                "instance_id": "inst-001",
            }
            if self.max_emulate_actions_items is not None:
                response["max_emulate_actions_items"] = self.max_emulate_actions_items
            return response

        if operation == "get_decision":
            decision = self.decisions[request["branch_id"]]
            return {
                **_common(request),
                "instance_id": request["instance_id"],
                "status": "completed",
                **decision,
            }

        if operation == "commit_action":
            return {
                **_common(request),
                "instance_id": request["instance_id"],
                "status": "completed",
                "branch_id": "root",
                "decision_point_id": "d-root-committed",
                "branch_log": [],
                "masked_emulator_dto": {
                    "mask_version": MASK_VERSION,
                    "legal_actions": [
                        {"action_id": "noop", "action_type": "system"}
                    ],
                },
            }

        if operation == "emulate_actions":
            self._emulate_actions_call_count += 1
            reject = self.reject_emulate_actions or (
                self.reject_emulate_actions_from_call is not None
                and self._emulate_actions_call_count >= self.reject_emulate_actions_from_call
            )
            if reject:
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

        if operation == "cancel_branches":
            self.cancel_calls.append(list(request["branch_ids"]))
            statuses = {bid: "cancelled" for bid in request["branch_ids"]}
            return {
                **_common(request),
                "instance_id": request["instance_id"],
                "status": "completed",
                "branch_statuses": statuses,
            }

        if operation == "release_branches":
            self.release_calls.append(list(request["branch_ids"]))
            statuses = {bid: "released" for bid in request["branch_ids"]}
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


def _root_decision(legal_actions: list[dict], *, decision_point_id: str = "d-root", **dto_extra) -> dict:
    return {
        "decision_point_id": decision_point_id,
        "masked_emulator_dto": {
            "mask_version": MASK_VERSION,
            "legal_actions": legal_actions,
            **dto_extra,
        },
    }


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


def _alive_dto(legal_actions: list[dict] | None = None) -> dict:
    dto = {
        "mask_version": MASK_VERSION,
        "hp": 40,
        "maxHp": 50,
        "enemies": [{"hp": 10, "maxHp": 10, "isAlive": True, "intent": {}}],
        "legal_actions": _COMBAT_ACTIONS,
    }
    if legal_actions is not None:
        dto["legal_actions"] = legal_actions
    return dto


_COMBAT_ACTIONS = [
    {"action_id": "strike", "action_type": "card", "is_available": True},
    {"action_id": "end", "action_type": "system", "is_available": True},
]


class BeamSearchEngineTest(unittest.IsolatedAsyncioTestCase):
    async def _client(self) -> tuple[AsyncTrainingApiClient, _FakeConnection]:
        connection = _FakeConnection()
        client = AsyncTrainingApiClient(connection)  # type: ignore[arg-type]
        await client.start_instance({"instance_type": "combat"}, timeout_s=1.0)
        return client, connection

    def _engine(
        self,
        client: AsyncTrainingApiClient,
        *,
        max_depth: int = 1,
        beam_width: int = 8,
        top_k_actions: int = 2,
        max_batch_size: int = 64,
    ) -> BeamSearchEngine:
        return BeamSearchEngine(
            client,
            policy=PriorHeuristicPolicy(),
            value_fn=HeuristicValueFunction(),
            config=BeamSearchConfig(
                max_depth=max_depth,
                top_k_actions=top_k_actions,
                beam_width=beam_width,
                max_batch_size=max_batch_size,
            ),
        )

    async def test_single_depth_prefers_the_winning_action(self) -> None:
        client, connection = await self._client()
        connection.emulate_results = {
            ("root", "strike"): {"status": "completed", "decision_point_id": "d-b1", "masked_emulator_dto": _victory_dto()},
            ("root", "end"): {"status": "completed", "decision_point_id": "d-b2", "masked_emulator_dto": _alive_dto()},
        }
        engine = self._engine(client, max_depth=1)

        result = await engine.search("inst-001", _root_decision(_COMBAT_ACTIONS), timeout_s=5.0)

        self.assertEqual(result.best_root_action_id, "strike")
        self.assertEqual(result.reason, "max_depth")
        self.assertEqual(result.stats.branches_created, 2)

    async def test_branch_fault_is_counted_even_when_a_sibling_branch_succeeds(self) -> None:
        # Before branches_faulted existed, a fault on one candidate ("strike") was
        # completely invisible whenever a sibling candidate ("end") still resolved -
        # the search finished normally with zero record that anything had failed.
        client, connection = await self._client()
        connection.emulate_results = {
            ("root", "strike"): {
                "status": "faulted",
                "error": "boom",
                "fault_kind": "action_fault",
            },
            ("root", "end"): {"status": "completed", "decision_point_id": "d-b2", "masked_emulator_dto": _alive_dto()},
        }
        engine = self._engine(client, max_depth=1)

        result = await engine.search("inst-001", _root_decision(_COMBAT_ACTIONS), timeout_s=5.0)

        self.assertEqual(result.best_root_action_id, "end")
        self.assertEqual(result.stats.branches_created, 2)
        self.assertEqual(result.stats.branches_faulted, 1)

    async def test_branch_fault_is_traced_with_its_status_and_fault_kind(self) -> None:
        client, connection = await self._client()
        connection.emulate_results = {
            ("root", "strike"): {
                "status": "faulted",
                "error": "boom",
                "fault_kind": "action_fault",
            },
            ("root", "end"): {"status": "completed", "decision_point_id": "d-b2", "masked_emulator_dto": _alive_dto()},
        }
        engine = self._engine(client, max_depth=1)
        collector = InMemorySearchTraceCollector()
        engine.trace_collector = collector

        await engine.search("inst-001", _root_decision(_COMBAT_ACTIONS), timeout_s=5.0)

        fault_events = [event for event in collector.events if isinstance(event, BranchFaultTrace)]
        self.assertEqual(len(fault_events), 1)
        fault = fault_events[0]
        self.assertEqual(fault.action_id, "strike")
        self.assertEqual(fault.root_action_id, "strike")
        self.assertEqual(fault.parent_branch_id, "root")
        self.assertEqual(fault.depth, 1)
        self.assertEqual(fault.combat_depth, 1)
        self.assertEqual(fault.continuation_steps, 0)
        self.assertEqual(fault.status, "faulted")
        self.assertEqual(fault.fault_kind, "action_fault")
        self.assertEqual(fault.detail, "boom")

    async def test_continuation_branch_fault_trace_uses_child_depth_coordinates(self) -> None:
        client, _connection = await self._client()
        engine = self._engine(client, max_depth=2)
        collector = InMemorySearchTraceCollector()
        engine.trace_collector = collector
        parent = BeamNode(
            branch_id="b-parent",
            parent_branch_id="root",
            rng_id=7,
            decision_point_id="d-choice",
            masked_emulator_dto={
                "mask_version": MASK_VERSION,
                "legal_actions": [
                    {"action_id": "pick", "action_type": "choice_card", "is_available": True}
                ],
            },
            depth=2,
            value=3.0,
            root_action_id="play-card",
            combat_depth=1,
            continuation_steps=2,
        )
        branch_results = {
            "b-child": {
                "status": "faulted",
                "error": "boom",
                "fault_kind": "action_fault",
            }
        }

        _next_beam, _finished, _ms, _depth, _limit, branches_faulted = engine._score_frontier(  # noqa: SLF001
            [(parent, ActionCandidate("pick"), "b-child", 7)],
            branch_results,
            search_id="s",
        )

        self.assertEqual(branches_faulted, 1)
        fault_events = [event for event in collector.events if isinstance(event, BranchFaultTrace)]
        self.assertEqual(len(fault_events), 1)
        fault = fault_events[0]
        self.assertEqual(fault.depth, 3)
        self.assertEqual(fault.combat_depth, 1)
        self.assertEqual(fault.continuation_steps, 3)

    async def test_multi_depth_finds_a_deferred_win(self) -> None:
        client, connection = await self._client()
        continue_actions = [
            {"action_id": "strike", "action_type": "card", "is_available": True},
            {"action_id": "end", "action_type": "system", "is_available": True},
        ]
        connection.emulate_results = {
            ("root", "strike"): {"status": "completed", "decision_point_id": "d-b1", "masked_emulator_dto": _victory_dto()},
            ("root", "end"): {
                "status": "completed",
                "decision_point_id": "d-b2",
                "masked_emulator_dto": _alive_dto(continue_actions),
            },
        }
        # Whatever branch_id RL assigns to the "root -> end" child, its own children
        # ("strike"/"end" again) key off of PARENT branch_id, not a fixed literal - so
        # populate lazily once we see the actual generated branch_id in the request.
        engine = self._engine(client, max_depth=2)

        # First depth resolves b1 (terminal victory) and b2 (alive); b2's own branch_id
        # is only known after the fact, so fill in its children's canned results here.
        original_emulate_actions = client.emulate_actions

        depth = {"n": 0}

        async def patched_emulate_actions(instance_id, items, *, timeout_s, simulation_options=None):
            depth["n"] += 1
            if depth["n"] == 2:
                for item in items:
                    connection.emulate_results.setdefault(
                        (item["parent_branch_id"], "strike"),
                        {"status": "completed", "decision_point_id": "d-c1", "masked_emulator_dto": _victory_dto()},
                    )
                    connection.emulate_results.setdefault(
                        (item["parent_branch_id"], "end"),
                        {"status": "completed", "decision_point_id": "d-c2", "masked_emulator_dto": _alive_dto()},
                    )
            return await original_emulate_actions(
                instance_id, items, timeout_s=timeout_s, simulation_options=simulation_options
            )

        client.emulate_actions = patched_emulate_actions  # type: ignore[assignment]

        result = await engine.search("inst-001", _root_decision(_COMBAT_ACTIONS), timeout_s=5.0)

        self.assertEqual(result.best_root_action_id, "strike")
        self.assertEqual(result.stats.depths_completed, 2)

    async def test_releases_all_created_branches_on_success(self) -> None:
        client, connection = await self._client()
        connection.emulate_results = {
            ("root", "strike"): {"status": "completed", "decision_point_id": "d-b1", "masked_emulator_dto": _victory_dto()},
            ("root", "end"): {"status": "completed", "decision_point_id": "d-b2", "masked_emulator_dto": _alive_dto()},
        }
        engine = self._engine(client, max_depth=1)

        await engine.search("inst-001", _root_decision(_COMBAT_ACTIONS), timeout_s=5.0)

        created_branch_ids = {
            item["branch_id"]
            for message in connection.messages
            if message["operation"] == "emulate_actions"
            for item in message["items"]
        }
        self.assertEqual(len(connection.cancel_calls), 1)
        self.assertEqual(len(connection.release_calls), 1)
        self.assertEqual(set(connection.cancel_calls[0]), created_branch_ids)
        self.assertEqual(set(connection.release_calls[0]), created_branch_ids)

    async def test_splits_wide_frontier_across_max_batch_size(self) -> None:
        # The max_branches capacity means a frontier wider than max_batch_size must be
        # sent as multiple same-depth emulate_actions requests, not one oversized one.
        three_actions = [
            {"action_id": "c1", "action_type": "card", "is_available": True},
            {"action_id": "c2", "action_type": "card", "is_available": True},
            {"action_id": "c3", "action_type": "card", "is_available": True},
        ]
        client, connection = await self._client()
        connection.emulate_results = {
            ("root", "c1"): {"status": "completed", "decision_point_id": "d-c1", "masked_emulator_dto": _alive_dto()},
            ("root", "c2"): {"status": "completed", "decision_point_id": "d-c2", "masked_emulator_dto": _victory_dto()},
            ("root", "c3"): {"status": "completed", "decision_point_id": "d-c3", "masked_emulator_dto": _alive_dto()},
        }
        engine = self._engine(client, max_depth=1, top_k_actions=3, max_batch_size=2)

        result = await engine.search("inst-001", _root_decision(three_actions), timeout_s=5.0)

        emulate_calls = [m for m in connection.messages if m["operation"] == "emulate_actions"]
        self.assertEqual(len(emulate_calls), 2)
        self.assertEqual([len(c["items"]) for c in emulate_calls], [2, 1])
        self.assertEqual(result.best_root_action_id, "c2")
        self.assertEqual(result.stats.branches_created, 3)

    async def test_chunk_failure_keeps_earlier_chunk_results(self) -> None:
        three_actions = [
            {"action_id": "c1", "action_type": "card", "is_available": True},
            {"action_id": "c2", "action_type": "card", "is_available": True},
            {"action_id": "c3", "action_type": "card", "is_available": True},
        ]
        client, connection = await self._client()
        connection.emulate_results = {
            ("root", "c1"): {"status": "completed", "decision_point_id": "d-c1", "masked_emulator_dto": _victory_dto()},
            ("root", "c2"): {"status": "completed", "decision_point_id": "d-c2", "masked_emulator_dto": _alive_dto()},
        }
        # Reject starting from the 2nd emulate_actions call (the 2nd chunk, [c3]) - the
        # 1st chunk ([c1, c2], containing the eventual winner c1) already succeeded.
        connection.reject_emulate_actions_from_call = 2
        engine = self._engine(client, max_depth=1, top_k_actions=3, max_batch_size=2)
        collector = InMemorySearchTraceCollector()
        engine.trace_collector = collector

        result = await engine.search("inst-001", _root_decision(three_actions), timeout_s=5.0)

        self.assertEqual(result.best_root_action_id, "c1")
        self.assertTrue(result.reason.startswith("emulate_actions_rejected"))
        self.assertEqual(result.stats.branches_created, 2)
        self.assertEqual(result.stats.branches_faulted, 0)
        self.assertFalse(any(isinstance(event, BranchFaultTrace) for event in collector.events))
        # Two emulate_actions requests were SENT (the 2nd rejected), but only the first
        # chunk's branches actually exist on RL's side - cleanup must target only those.
        emulate_calls = [m for m in connection.messages if m["operation"] == "emulate_actions"]
        self.assertEqual(len(emulate_calls), 2)
        first_chunk_branch_ids = {item["branch_id"] for item in emulate_calls[0]["items"]}
        self.assertEqual(len(first_chunk_branch_ids), 2)
        self.assertEqual(set(connection.cancel_calls[0]), first_chunk_branch_ids)

    async def test_rejected_batch_falls_back_to_no_candidate(self) -> None:
        client, connection = await self._client()
        connection.reject_emulate_actions = True
        engine = self._engine(client, max_depth=1)
        collector = InMemorySearchTraceCollector()
        engine.trace_collector = collector

        result = await engine.search("inst-001", _root_decision(_COMBAT_ACTIONS), timeout_s=5.0)

        self.assertIsNone(result.best_root_action_id)
        self.assertTrue(result.reason.startswith("emulate_actions_rejected"))
        self.assertEqual(result.stats.branches_created, 0)
        self.assertEqual(result.stats.branches_faulted, 0)
        self.assertFalse(any(isinstance(event, BranchFaultTrace) for event in collector.events))
        # A rejected batch admits no Branches on RL's side - nothing to clean up.
        self.assertEqual(connection.cancel_calls, [])
        self.assertEqual(connection.release_calls, [])

    async def test_non_combat_decision_is_not_beam_searched(self) -> None:
        client, connection = await self._client()
        engine = self._engine(client, max_depth=1)
        non_combat_actions = [{"action_id": "opt-1", "action_type": "choice_reward_card", "is_available": True}]

        result = await engine.search("inst-001", _root_decision(non_combat_actions), timeout_s=5.0)

        self.assertIsNone(result.best_root_action_id)
        self.assertEqual(result.reason, "not_beam_searchable")
        self.assertEqual([m["operation"] for m in connection.messages], ["start_instance"])

    async def test_legacy_whole_run_without_emulate_actions_capability_is_not_beam_searched(self) -> None:
        # Older Whole Run servers may omit max_emulate_actions_items. Combat instances
        # require it during protocol validation, so use whole_run here to exercise Beam's
        # compatibility guard through the real AsyncTrainingApiClient.
        connection = _FakeConnection()
        connection.max_emulate_actions_items = None
        client = AsyncTrainingApiClient(connection)  # type: ignore[arg-type]
        await client.start_instance({"instance_type": "whole_run"}, timeout_s=1.0)
        engine = self._engine(client, max_depth=1)

        result = await engine.search("inst-001", _root_decision(_COMBAT_ACTIONS), timeout_s=5.0)

        self.assertIsNone(result.best_root_action_id)
        self.assertEqual(result.reason, "emulate_actions_not_supported")
        self.assertEqual([m["operation"] for m in connection.messages], ["start_instance"])

    async def test_no_legal_actions_short_circuits(self) -> None:
        client, connection = await self._client()
        engine = self._engine(client, max_depth=1)

        result = await engine.search(
            "inst-001", _root_decision([], decision_point_id="d-root"), timeout_s=5.0
        )

        self.assertEqual(result.reason, "no_legal_actions")


if __name__ == "__main__":
    unittest.main()