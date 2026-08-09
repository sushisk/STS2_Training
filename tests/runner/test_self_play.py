"""Coverage for the self-play batch driver and CLI."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sts2_training.runner.episode import EpisodeResult
from sts2_training.runner.self_play import (
    SelfPlayRunResult,
    _parse_args,
    _summarize,
    main,
    run_self_play_batch,
)
from sts2_training.selection_log import JsonlSelectionLogger

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
    """One decision, then terminal victory; optional start/close failures."""

    def __init__(self, *, fail_start: bool = False, fail_close: bool = False) -> None:
        self.client_session_id = f"session-{id(self)}"
        self._fail_start = fail_start
        self._fail_close = fail_close
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
                "masked_emulator_dto": {
                    "legal_actions": [],
                    "terminal": True,
                    "outcome": "victory",
                },
            }
        if operation == "close_instance":
            return {
                **_common(request),
                "instance_id": request["instance_id"],
                "status": "completed",
            }

        raise AssertionError(f"unexpected operation: {operation}")

    async def invalidate(self) -> None:
        pass

    async def close(self) -> None:
        if self._fail_close:
            raise OSError("close failed")


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

    async def test_logger_creation_failures_are_per_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "not-a-directory"
            output_dir.write_text("occupied", encoding="utf-8")

            results = await run_self_play_batch(
                character_id="IRONCLAD",
                num_runs=2,
                concurrency=2,
                connection_factory=lambda: _FakeConnection(),
                output_dir=output_dir,
            )

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.episode is None for result in results))
        self.assertTrue(all("FileExistsError" in (result.error or "") for result in results))

    async def test_logger_write_failure_marks_only_that_run_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                JsonlSelectionLogger, "__call__", side_effect=OSError("disk full")
            ):
                results = await run_self_play_batch(
                    character_id="IRONCLAD",
                    num_runs=1,
                    connection_factory=lambda: _FakeConnection(),
                    output_dir=Path(tmp),
                )

        self.assertIsNone(results[0].episode)
        self.assertIn("SelectionLogError: OSError: disk full", results[0].error or "")

    async def test_logger_close_failure_marks_only_that_run_failed(self) -> None:
        original_close = JsonlSelectionLogger.close

        def close_then_fail(logger: JsonlSelectionLogger) -> None:
            original_close(logger)
            raise OSError("flush failed")

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(JsonlSelectionLogger, "close", new=close_then_fail):
                results = await run_self_play_batch(
                    character_id="IRONCLAD",
                    num_runs=1,
                    connection_factory=lambda: _FakeConnection(),
                    output_dir=Path(tmp),
                )

        self.assertIsNone(results[0].episode)
        self.assertIn("SelectionLogCloseError: OSError: flush failed", results[0].error or "")

    async def test_connection_close_failure_does_not_overwrite_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = await run_self_play_batch(
                character_id="IRONCLAD",
                num_runs=1,
                connection_factory=lambda: _FakeConnection(fail_close=True),
                output_dir=Path(tmp),
            )

        self.assertIsNone(results[0].error)
        self.assertIsNotNone(results[0].episode)

    async def test_character_id_is_sanitized_for_filename_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            results = await run_self_play_batch(
                character_id="IRON/CLAD: A20",
                num_runs=1,
                connection_factory=lambda: _FakeConnection(),
                output_dir=output_dir,
            )

            self.assertEqual(results[0].log_path.parent, output_dir)
            self.assertTrue(results[0].log_path.name.startswith("iron-clad-a20-"))

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
            results = await run_self_play_batch(
                character_id="IRONCLAD",
                num_runs=6,
                concurrency=2,
                connection_factory=lambda: _TrackingConnection(),
                decision_timeout_s=5.0,
                output_dir=Path(tmp),
            )

        self.assertEqual(len(results), 6)
        self.assertLessEqual(max_active, 2)

    async def test_shared_invalid_config_is_rejected_before_connection_factory(self) -> None:
        calls = 0

        def factory() -> _FakeConnection:
            nonlocal calls
            calls += 1
            return _FakeConnection()

        invalid_overrides = (
            {"num_runs": 0},
            {"concurrency": 0},
            {"decision_timeout_s": 0},
            {"max_decisions": 0},
            {"beam_max_depth": 0},
            {"search_mode": "nonexistent"},
            {"character_id": ""},
            {"ascension": "bad"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            for override in invalid_overrides:
                with self.subTest(override=override):
                    base = {
                        "character_id": "IRONCLAD",
                        "num_runs": 1,
                        "concurrency": 1,
                        "connection_factory": factory,
                        "output_dir": Path(tmp),
                    }
                    with self.assertRaises(ValueError):
                        await run_self_play_batch(**{**base, **override})

        self.assertEqual(calls, 0)


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
        self.assertEqual(
            summary["failures"],
            [{"run_id": "r2", "log_path": "r2.jsonl", "error": "RuntimeError: boom"}],
        )

    def test_empty_results_do_not_divide_by_zero(self) -> None:
        summary = _summarize([])

        self.assertEqual(summary["runs_requested"], 0)
        self.assertIsNone(summary["avg_decisions_made"])
        self.assertIsNone(summary["avg_elapsed_s"])


class SelfPlayCliTest(unittest.TestCase):
    def test_parse_args_wires_self_play_specific_options(self) -> None:
        args = _parse_args(
            [
                "--character-id",
                "IRONCLAD",
                "--num-runs",
                "7",
                "--concurrency",
                "3",
                "--output-dir",
                "custom-data",
                "--search-mode",
                "deep",
            ]
        )

        self.assertEqual(args.character_id, "IRONCLAD")
        self.assertEqual(args.num_runs, 7)
        self.assertEqual(args.concurrency, 3)
        self.assertEqual(args.output_dir, Path("custom-data"))
        self.assertEqual(args.search_mode, "deep")

    def test_main_prints_summary_and_uses_failure_exit_code(self) -> None:
        cases = (
            (
                [
                    SelfPlayRunResult(
                        "ok",
                        Path("ok.jsonl"),
                        EpisodeResult("i", 1, {"outcome": "victory"}, 1.0),
                        None,
                    )
                ],
                0,
            ),
            ([SelfPlayRunResult("bad", Path("bad.jsonl"), None, "RuntimeError: boom")], 1),
        )
        for results, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                fake_run = mock.AsyncMock(return_value=results)
                stdout = io.StringIO()
                with (
                    mock.patch("sts2_training.runner.self_play._run", fake_run),
                    mock.patch(
                        "sts2_training.runner.self_play.configure_logging"
                    ) as configure_logging,
                    contextlib.redirect_stdout(stdout),
                ):
                    code = main(
                        [
                            "--character-id",
                            "IRONCLAD",
                            "--num-runs",
                            "1",
                            "--log-level",
                            "INFO",
                        ]
                    )

                self.assertEqual(code, expected_code)
                configure_logging.assert_called_once_with("INFO")
                fake_run.assert_awaited_once()
                payload = json.loads(stdout.getvalue())
                self.assertEqual(payload["runs_failed"], expected_code)


if __name__ == "__main__":
    unittest.main()
