"""Coverage for `EpisodeRunner`'s decide+commit loop, against a scripted fake RL
connection. Every decision here carries exactly one legal action, so
`CombatDecisionEngine.decide()` always takes the `forced_single_action` path (see
`engine.py`) - this test is about the loop/lifecycle around decisions, not about
beam search itself (already covered by `tests/decision/test_beam_search.py`).
"""

from __future__ import annotations

import unittest

from sts2_training.api.async_client import AsyncTrainingApiClient
from sts2_training.decision.engine import CombatDecisionEngine
from sts2_training.decision.search_modes import SEARCH_MODES
from sts2_training.runner.episode import EpisodeLimitExceeded, EpisodeRunner, build_engine

_ACTION = {"action_id": "a", "action_type": "system", "is_available": True}


def _common(request: dict) -> dict:
    return {
        "schema_version": "0.7",
        "server_epoch": "epoch-1",
        "client_session_id": request["client_session_id"],
        "request_seq": request["request_seq"],
        "request_id": request["request_id"],
        "operation": request["operation"],
    }


class _FakeConnection:
    """`decisions[0]` is what `get_decision` returns before any commit;
    `decisions[i + 1]` is what the i-th `commit_action` returns (and what a
    subsequent `get_decision` would also see, tracked via `_index`)."""

    client_session_id = "session-a"

    def __init__(self, decisions: list[dict]) -> None:
        self._decisions = decisions
        self._index = 0
        self.close_instance_calls = 0
        self.close_instance_instance_ids: list[str] = []

    async def exchange(self, message: dict, *, deadline: float) -> dict:
        request = dict(message)
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
                **self._decisions[self._index],
            }
        if operation == "commit_action":
            self._index += 1
            return {
                **_common(request),
                "instance_id": request["instance_id"],
                "status": "completed",
                "branch_id": "root",
                **self._decisions[self._index],
            }
        if operation == "close_instance":
            self.close_instance_calls += 1
            self.close_instance_instance_ids.append(request["instance_id"])
            return {**_common(request), "instance_id": request["instance_id"], "status": "completed"}

        raise AssertionError(f"unexpected operation: {operation}")

    async def invalidate(self) -> None:
        pass

    async def close(self) -> None:
        pass


def _decision(decision_point_id: str, *, legal_actions=None, **dto_extra) -> dict:
    return {
        "decision_point_id": decision_point_id,
        "masked_emulator_dto": {"legal_actions": legal_actions if legal_actions is not None else [_ACTION], **dto_extra},
    }


class EpisodeRunnerTest(unittest.IsolatedAsyncioTestCase):
    async def _client_and_connection(self, decisions: list[dict]) -> tuple[AsyncTrainingApiClient, _FakeConnection]:
        connection = _FakeConnection(decisions)
        client = AsyncTrainingApiClient(connection)  # type: ignore[arg-type]
        await client.start_instance({"instance_type": "combat"}, timeout_s=1.0)
        return client, connection

    async def test_runs_until_terminal_and_closes_instance(self) -> None:
        decisions = [
            _decision("d0"),
            _decision("d1"),
            _decision("d2", legal_actions=[], outcome="victory"),
        ]
        client, connection = await self._client_and_connection(decisions)
        runner = EpisodeRunner(client)

        result = await runner.run("inst-001", decision_timeout_s=5.0)

        self.assertEqual(result.decisions_made, 2)
        self.assertEqual(result.decision_sources, {"forced_single_action": 2})
        self.assertEqual(result.final_dto["outcome"], "victory")
        self.assertEqual(result.final_dto["legal_actions"], [])
        self.assertEqual(connection.close_instance_calls, 1)
        self.assertEqual(connection.close_instance_instance_ids, ["inst-001"])

    async def test_already_terminal_at_start_makes_no_commit_but_reports_final_dto(self) -> None:
        decisions = [_decision("d0", legal_actions=[], outcome="defeat")]
        client, connection = await self._client_and_connection(decisions)
        runner = EpisodeRunner(client)

        result = await runner.run("inst-001", decision_timeout_s=5.0)

        self.assertEqual(result.decisions_made, 0)
        self.assertEqual(result.decision_sources, {"none": 1})
        self.assertEqual(result.final_dto["outcome"], "defeat")
        self.assertEqual(connection.close_instance_calls, 1)

    async def test_max_decisions_raises_but_still_closes_instance(self) -> None:
        decisions = [_decision("d0"), _decision("d1"), _decision("d2"), _decision("d3", legal_actions=[])]
        client, connection = await self._client_and_connection(decisions)
        runner = EpisodeRunner(client)

        with self.assertRaises(EpisodeLimitExceeded):
            await runner.run("inst-001", decision_timeout_s=5.0, max_decisions=2)

        self.assertEqual(connection.close_instance_calls, 1)

    async def test_close_instance_skipped_when_session_invalid(self) -> None:
        # `_close_best_effort` is exercised directly against a minimal stub rather than
        # through `run()`, since a real AsyncTrainingApiClient with `session_invalid`
        # already set rejects every operation up front (including get_decision), never
        # reaching the close step this test is actually about.
        class _StubClient:
            session_invalid = True
            pending_retry = None

            def __init__(self) -> None:
                self.close_instance_calls = 0

            async def close_instance(self, instance_id: str, *, timeout_s: float) -> dict:
                self.close_instance_calls += 1
                return {"status": "completed"}

        client = _StubClient()
        runner = EpisodeRunner(client)  # type: ignore[arg-type]

        await runner._close_best_effort("inst-001", 5.0)  # noqa: SLF001

        self.assertEqual(client.close_instance_calls, 0)


class BuildEngineTest(unittest.TestCase):
    def test_no_args_builds_a_default_engine(self) -> None:
        engine = build_engine(client=object())

        self.assertIsInstance(engine, CombatDecisionEngine)

    def test_search_mode_and_beam_max_depth_configure_the_built_engine(self) -> None:
        engine = build_engine(client=object(), search_mode="deep", beam_max_depth=9)

        self.assertEqual(engine.beam_search.config.max_depth, 9)

    def test_named_mode_alone_is_applied(self) -> None:
        engine = build_engine(client=object(), search_mode="wide")

        self.assertEqual(engine.beam_search.config.beam_width, SEARCH_MODES["wide"].beam_width)

    def test_explicit_engine_is_returned_as_is(self) -> None:
        given = CombatDecisionEngine(client=object())

        self.assertIs(build_engine(client=object(), engine=given), given)

    def test_engine_combined_with_search_mode_raises(self) -> None:
        given = CombatDecisionEngine(client=object())

        with self.assertRaises(ValueError):
            build_engine(client=object(), engine=given, search_mode="deep")

    def test_engine_combined_with_beam_max_depth_raises(self) -> None:
        given = CombatDecisionEngine(client=object())

        with self.assertRaises(ValueError):
            build_engine(client=object(), engine=given, beam_max_depth=3)


if __name__ == "__main__":
    unittest.main()
