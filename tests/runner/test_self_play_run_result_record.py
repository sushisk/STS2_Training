from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sts2_training.runner.self_play import run_self_play_batch

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


class _Connection:
    client_session_id = "session-test"

    def __init__(self) -> None:
        self._committed = False

    async def connect(self) -> None:
        pass

    async def exchange(self, message: dict, *, deadline: float) -> dict:
        request = dict(message)
        operation = request["operation"]

        if operation == "start_instance":
            return {
                **_common(request),
                "status": "completed",
                "instance_id": "instance-1",
            }
        if operation == "get_decision":
            dto = (
                {"legal_actions": [], "run_terminal": True, "outcome": "victory"}
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
                    "run_terminal": True,
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
        pass


class SelfPlayRunResultRecordTest(unittest.IsolatedAsyncioTestCase):
    async def test_successful_run_keeps_selection_and_appends_terminal_outcome_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "sts2_training.runner.self_play.random.randint",
                return_value=123456,
            ):
                results = await run_self_play_batch(
                    character_id="IRONCLAD",
                    ascension=3,
                    num_runs=1,
                    connection_factory=_Connection,
                    output_dir=Path(tmp),
                )

            self.assertEqual(len(results), 1)
            self.assertIsNone(results[0].error)
            records = [
                json.loads(line)
                for line in results[0].log_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(records), 2)
        selection, terminal = records
        self.assertEqual(selection["event"], "selection")
        self.assertEqual(selection["request"]["operation"], "commit_action")
        self.assertEqual(selection["run_result"], "victory")

        self.assertEqual(terminal["event"], "self_play_run_result")
        self.assertEqual(terminal["run_id"], results[0].run_id)
        self.assertEqual(terminal["seed"], 123456)
        self.assertEqual(terminal["character_id"], "IRONCLAD")
        self.assertEqual(terminal["ascension"], 3)
        self.assertEqual(terminal["instance_id"], "instance-1")
        self.assertEqual(terminal["decisions_made"], 1)
        self.assertEqual(terminal["outcome"], "victory")
        self.assertEqual(
            terminal["final_dto"],
            {"legal_actions": [], "run_terminal": True, "outcome": "victory"},
        )


if __name__ == "__main__":
    unittest.main()
