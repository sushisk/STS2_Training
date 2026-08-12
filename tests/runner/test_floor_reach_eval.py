"""Coverage for the floor-reach evaluation harness (no live server required)."""

from __future__ import annotations

import unittest

from sts2_training.runner.floor_reach_eval import (
    FloorReachResult,
    run_floor_reach_eval,
    summarize_floor_reach,
)

_CARD_ACTION = {
    "action_id": "a-card",
    "action_type": "card",
    "is_available": True,
    "parameters": {"cardId": "STRIKE_IRONCLAD"},
}


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
    """Two decisions climbing totalFloor 1 -> 3, then terminal defeat."""

    def __init__(self, *, fail_start: bool = False) -> None:
        self.client_session_id = f"session-{id(self)}"
        self._fail_start = fail_start
        self._step = 0

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
            dto = self._dto_for_step()
            return {
                **_common(request),
                "instance_id": request["instance_id"],
                "status": "completed",
                "branch_id": "root",
                "decision_point_id": f"d{self._step}",
                "masked_emulator_dto": dto,
            }
        if operation == "commit_action":
            self._step += 1
            dto = self._dto_for_step()
            return {
                **_common(request),
                "instance_id": request["instance_id"],
                "status": "completed",
                "branch_id": "root",
                "decision_point_id": f"d{self._step}",
                "masked_emulator_dto": dto,
            }
        if operation == "close_instance":
            return {**_common(request), "instance_id": request["instance_id"], "status": "completed"}

        raise AssertionError(f"unexpected operation: {operation}")

    def _dto_for_step(self) -> dict:
        if self._step >= 2:
            return {
                "legal_actions": [],
                "terminal": True,
                "outcome": "defeat",
                "totalFloor": 3,
                "currentActIndex": 0,
            }
        return {
            "legal_actions": [_CARD_ACTION],
            "totalFloor": 1 + self._step,
            "currentActIndex": 0,
        }

    async def invalidate(self) -> None:
        pass

    async def close(self) -> None:
        pass


class RunFloorReachEvalTest(unittest.IsolatedAsyncioTestCase):
    async def test_tracks_max_total_floor_across_decisions(self) -> None:
        results = await run_floor_reach_eval(
            character_id="IRONCLAD",
            num_runs=1,
            concurrency=1,
            use_beam=False,
            connection_factory=lambda: _FakeConnection(),
            decision_timeout_s=5.0,
        )

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertIsNone(result.error)
        self.assertEqual(result.max_total_floor, 3)
        self.assertEqual(result.outcome, "defeat")
        self.assertGreaterEqual(result.decisions_made, 1)

    async def test_concurrent_runs_do_not_cross_contaminate_floor_tracking(self) -> None:
        # Exercises the ContextVar isolation: many concurrent runs, same fixed per-run
        # trajectory (floor 1 -> 3) - if state leaked between concurrent Tasks, some
        # results would show an inflated or zeroed max_total_floor.
        results = await run_floor_reach_eval(
            character_id="IRONCLAD",
            num_runs=8,
            concurrency=8,
            use_beam=False,
            connection_factory=lambda: _FakeConnection(),
            decision_timeout_s=5.0,
        )

        self.assertEqual(len(results), 8)
        for result in results:
            self.assertIsNone(result.error)
            self.assertEqual(result.max_total_floor, 3)

    async def test_one_failing_run_does_not_sink_the_batch(self) -> None:
        connections = [_FakeConnection(), _FakeConnection(fail_start=True), _FakeConnection()]
        connections_iter = iter(connections)

        results = await run_floor_reach_eval(
            character_id="IRONCLAD",
            num_runs=3,
            concurrency=3,
            use_beam=False,
            connection_factory=lambda: next(connections_iter),
            decision_timeout_s=5.0,
        )

        failed = [r for r in results if r.error is not None]
        completed = [r for r in results if r.error is None]
        self.assertEqual(len(failed), 1)
        self.assertEqual(len(completed), 2)


class SummarizeFloorReachTest(unittest.TestCase):
    def _result(self, floor: int, *, error: "str | None" = None) -> FloorReachResult:
        return FloorReachResult(
            run_id="r",
            seed=1,
            max_total_floor=floor,
            act_index_at_max=0,
            decisions_made=1,
            decision_source_counts={},
            outcome="defeat" if error is None else None,
            error=error,
            elapsed_s=1.0,
        )

    def test_computes_mean_median_min_max(self) -> None:
        results = [self._result(f) for f in (2, 4, 6, 8)]
        summary = summarize_floor_reach(results)

        self.assertEqual(summary["floor_stats"]["mean"], 5.0)
        self.assertEqual(summary["floor_stats"]["median"], 5.0)
        self.assertEqual(summary["floor_stats"]["min"], 2)
        self.assertEqual(summary["floor_stats"]["max"], 8)
        self.assertEqual(summary["runs_errored"], 0)

    def test_errored_runs_still_counted_but_flagged(self) -> None:
        results = [self._result(5), self._result(0, error="TimeoutError: boom")]
        summary = summarize_floor_reach(results)

        self.assertEqual(summary["runs_requested"], 2)
        self.assertEqual(summary["runs_errored"], 1)
        self.assertEqual(len(summary["errors"]), 1)


if __name__ == "__main__":
    unittest.main()
