"""Coverage for the self-play batch driver: bounded concurrency, one log file per
run, a failing run not sinking the batch, and the summary counters. Uses a fake
connection (same shape as `tests/runner/test_episode.py`'s `_FakeConnection`)
rather than a real TCP server - `start_new_run`/`EpisodeRunner` internals are
already covered elsewhere.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from sts2_training.runner.episode import EpisodeResult
from sts2_training.runner.self_play import (
    SelfPlayRunResult,
    _summarize,
    run_self_play_batch,
)

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
    """One decision, then terminal victory - enough to exercise the batch driver
    without re-testing `EpisodeRunner`'s loop itself. `fail_start` makes
    `start_instance` come back rejected, the way a real bad request would."""

    def __init__(self, *, fail_start: bool = False) -> None:
        self.client_session_id = f"session-{id(self)}"
        self._fail_start = fail_start
        self._committed = False

    async def connect(self) -> None:
        pass

    async def exchange(self, message: dict, *, deadline: float) -> dict:
        request = dict(message)
        operation = request["operation"]

        if operation == "start_instance":
            if self._fail_start:
                return {**_common(request), "status": "rejected", "fault_kind": "invalid_request"}
            return {
                **_common(request),
                "status": "completed",
                "instance_id": "inst-001",
                "max_emulate_actions_items": 8,
            }
        if operation == "get_decision":
            dto = (
                {"legal_actions": [], "terminal": True, "outcome": "victory"}
                if self._committed
                else {"legal_actions": [_ACTION]}
            )
            return {
                **_common(request),
                "instance_id": request["instance_id"],
                "status": "completed",
                "branch_id": "root",
                "decision_point_id": "d1" if self._committed else "d0",
                "masked_emulator_dto": dto,
            }
        if operation == "commit_action":
            self._committed = True
            return {
                **_common(request),
                "instance_id": request["instance_id"],
                "status": "completed",
                "branch_id": "root",
                "decision_point_id": "d1",
                "masked_emulator_dto": {"legal_actions": [], "terminal": True, "outcome": "victory"},
            }
        if operation == "close_instance":
            return {**_common(request), "instance_id": request["instance_id"], "status": "completed"}

        raise AssertionError(f"unexpected operation: {operation}")

    async def invalidate(self) -> None:
        pass

    async def close(self) -> None:
        pass


class RunSelfPlayBatchTest(unittest.IsolatedAsyncioTestCase):
    async def test_runs_all_and_writes_one_log_file_per_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            results = await run_self_play_batch(
                character_id="IRONCLAD",
                num_runs=3,
                concurrency=2,
                connection_factory=lambda: _FakeConnection(),
                decision_timeout_s=5.0,
                output_dir=output_dir,
            )

            self.assertEqual(len(results), 3)
            log_paths = {result.log_path for result in results}
            self.assertEqual(len(log_paths), 3)
            for result in results:
                self.assertIsNone(result.error)
                self.assertIsNotNone(result.episode)
                self.assertEqual(result.episode.decisions_made, 1)
                self.assertEqual(result.episode.final_dto["outcome"], "victory")
                self.assertTrue(result.log_path.exists())
                self.assertGreater(result.log_path.stat().st_size, 0)

    async def test_one_failing_run_does_not_sink_the_batch(self) -> None:
        connections = [
            _FakeConnection(fail_start=False),
            _FakeConnection(fail_start=True),
            _FakeConnection(fail_start=False),
        ]
        connections_iter = iter(connections)

        with tempfile.TemporaryDirectory() as tmp:
            results = await run_self_play_batch(
                character_id="IRONCLAD",
                num_runs=3,
                concurrency=3,
                connection_factory=lambda: next(connections_iter),
                decision_timeout_s=5.0,
                output_dir=Path(tmp),
            )

        failed = [result for result in results if result.error is not None]
        completed = [result for result in results if result.error is None]
        self.assertEqual(len(failed), 1)
        self.assertEqual(len(completed), 2)
        self.assertIsNone(failed[0].episode)

    async def test_concurrency_bounds_simultaneous_runs(self) -> None:
        active = 0
        max_active = 0
        lock = asyncio.Lock()

        class _TrackingConnection(_FakeConnection):
            async def exchange(self, message: dict, *, deadline: float) -> dict:
                nonlocal active, max_active
                if message["operation"] == "start_instance":
                    async with lock:
                        active += 1
                        max_active = max(max_active, active)
                    await asyncio.sleep(0.01)
                    async with lock:
                        active -= 1
                return await super().exchange(message, deadline=deadline)

        with tempfile.TemporaryDirectory() as tmp:
            await run_self_play_batch(
                character_id="IRONCLAD",
                num_runs=6,
                concurrency=2,
                connection_factory=lambda: _TrackingConnection(),
                decision_timeout_s=5.0,
                output_dir=Path(tmp),
            )

        self.assertLessEqual(max_active, 2)

    async def test_non_positive_num_runs_or_concurrency_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for kwargs in ({"num_runs": 0}, {"num_runs": -1}, {"concurrency": 0}, {"concurrency": -1}):
                with self.subTest(kwargs=kwargs):
                    base = {
                        "character_id": "IRONCLAD",
                        "num_runs": 1,
                        "concurrency": 1,
                        "connection_factory": lambda: _FakeConnection(),
                        "output_dir": Path(tmp),
                    }
                    with self.assertRaises(ValueError):
                        await run_self_play_batch(**{**base, **kwargs})


class SummarizeTest(unittest.TestCase):
    def test_counts_outcomes_and_failures(self) -> None:
        ok_episode = EpisodeResult(
            instance_id="i1", decisions_made=2, final_dto={"outcome": "victory"}, elapsed_s=1.5
        )
        results = [
            SelfPlayRunResult("r1", Path("r1.jsonl"), ok_episode, None),
            SelfPlayRunResult("r2", Path("r2.jsonl"), None, "RuntimeError: boom"),
        ]

        summary = _summarize(results)

        self.assertEqual(summary["runs_requested"], 2)
        self.assertEqual(summary["runs_completed"], 1)
        self.assertEqual(summary["runs_failed"], 1)
        self.assertEqual(summary["outcome_counts"], {"victory": 1})
        self.assertEqual(summary["avg_decisions_made"], 2.0)
        self.assertEqual(summary["failures"], [{"run_id": "r2", "error": "RuntimeError: boom"}])

    def test_empty_results_do_not_divide_by_zero(self) -> None:
        summary = _summarize([])

        self.assertEqual(summary["runs_requested"], 0)
        self.assertIsNone(summary["avg_decisions_made"])
        self.assertIsNone(summary["avg_elapsed_s"])


if __name__ == "__main__":
    unittest.main()
