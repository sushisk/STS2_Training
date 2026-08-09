from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sts2_training.runner.episode import EpisodeResult
from sts2_training.runner.self_play import run_self_play_batch


class _Connection:
    client_session_id = "session-test"

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass


class SelfPlayRunResultRecordTest(unittest.IsolatedAsyncioTestCase):
    async def test_successful_run_appends_terminal_outcome_record(self) -> None:
        episode = EpisodeResult(
            instance_id="instance-1",
            decisions_made=7,
            final_dto={"terminal": True, "outcome": "victory"},
            elapsed_s=1.25,
        )

        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch("sts2_training.runner.self_play.random.randint", return_value=123456),
                mock.patch(
                    "sts2_training.runner.self_play.CombatDecisionEngine",
                    return_value=object(),
                ),
                mock.patch(
                    "sts2_training.runner.self_play.start_new_run",
                    new=mock.AsyncMock(return_value=episode),
                ),
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

        self.assertEqual(len(records), 1)
        terminal = records[0]
        self.assertEqual(terminal["event"], "self_play_run_result")
        self.assertEqual(terminal["run_id"], results[0].run_id)
        self.assertEqual(terminal["seed"], 123456)
        self.assertEqual(terminal["character_id"], "IRONCLAD")
        self.assertEqual(terminal["ascension"], 3)
        self.assertEqual(terminal["instance_id"], "instance-1")
        self.assertEqual(terminal["decisions_made"], 7)
        self.assertEqual(terminal["elapsed_s"], 1.25)
        self.assertEqual(terminal["outcome"], "victory")
        self.assertEqual(terminal["final_dto"], {"terminal": True, "outcome": "victory"})


if __name__ == "__main__":
    unittest.main()
