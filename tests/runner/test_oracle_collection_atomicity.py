from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sts2_training.runner.oracle_collection import OracleEpisodeRunner


class _Client:
    def __init__(self) -> None:
        self.closed: list[str] = []

    async def get_decision(self, instance_id, branch_id, *, timeout_s):
        return {
            "decision_point_id": "d-root",
            "masked_emulator_dto": {
                "legal_actions": [
                    {"action_id": "a", "action_type": "card", "is_available": True}
                ]
            },
        }

    async def commit_action(self, instance_id, decision_point_id, action_id, *, timeout_s):
        return {
            "decision_point_id": "d-terminal",
            "masked_emulator_dto": {
                "terminal": True,
                "outcome": "victory",
                "legal_actions": [],
            },
        }

    async def close_instance(self, instance_id, *, timeout_s):
        self.closed.append(instance_id)


class _Oracle:
    async def collect(self, instance_id, decision, *, timeout_s):
        return object()


class _CommitEngine:
    def __init__(self, client) -> None:
        self.client = client

    async def decide(self, instance_id, *, timeout_s, decision):
        return SimpleNamespace(chosen_action_id="a", source="test", beam_result=None)


class _PartialFailingWriter:
    """Simulate an I/O failure after part of the episode-result JSON was appended."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, *args, **kwargs):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write('{"record_type":"combat_oracle_decision"}\n')

    def write_episode_result(self, *args, **kwargs):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write('{"record_type":"combat_oracle_episode_result"')
            handle.flush()
        raise OSError("episode result append failed")


class OracleEpisodeAtomicityTest(unittest.IsolatedAsyncioTestCase):
    async def test_episode_result_append_failure_rolls_back_entire_episode(self) -> None:
        client = _Client()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle.jsonl"
            prefix = '{"record_type":"previous_complete_episode"}\n'
            path.write_text(prefix, encoding="utf-8")
            runner = OracleEpisodeRunner(
                client,
                oracle=_Oracle(),  # type: ignore[arg-type]
                commit_engine=_CommitEngine(client),  # type: ignore[arg-type]
                writer=_PartialFailingWriter(path),  # type: ignore[arg-type]
            )

            with self.assertRaisesRegex(OSError, "episode result append failed"):
                await runner.run(
                    "inst",
                    oracle_timeout_s=5.0,
                    decision_timeout_s=5.0,
                )

            output = path.read_text(encoding="utf-8")

        self.assertEqual(output, prefix)
        self.assertEqual(client.closed, ["inst"])


if __name__ == "__main__":
    unittest.main()
