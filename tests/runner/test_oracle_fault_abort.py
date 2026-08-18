from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts2_training.decision.beam_search import BranchFaultAbortError, BranchFaultSummary
from sts2_training.decision.oracle_log import (
    ORACLE_EPISODE_RESULT_SCHEMA_VERSION,
    ORACLE_VALUE_MASK_VERSION,
    OracleJsonlWriter,
)
from sts2_training.runner.oracle_collection import OracleEpisodeRunner


_DTO_VERSION = "emulator-test"


class _Client:
    pending_retry = None
    session_invalid = False

    def __init__(self) -> None:
        self.commits: list[str] = []
        self.closed: list[str] = []

    async def get_decision(self, instance_id, branch_id, *, timeout_s):
        del instance_id, branch_id, timeout_s
        return {
            "status": "completed",
            "server_epoch": "epoch-1",
            "decision_point_id": "d-root",
            "masked_emulator_dto": {
                "mask_version": ORACLE_VALUE_MASK_VERSION,
                "dto_version": _DTO_VERSION,
                "legal_actions": [
                    {"action_id": "end", "action_type": "system", "is_available": True}
                ],
            },
        }

    async def commit_action(self, instance_id, decision_point_id, action_id, *, timeout_s):
        del instance_id, decision_point_id, timeout_s
        self.commits.append(action_id)
        raise AssertionError("commit_action must not run after Oracle branch-fault abort")

    async def close_instance(self, instance_id, *, timeout_s):
        del timeout_s
        self.closed.append(instance_id)


class _CommitEngine:
    def __init__(self, client) -> None:
        self.client = client
        self.decisions: list[str] = []

    async def decide(self, instance_id, *, timeout_s, decision):
        del instance_id, timeout_s
        self.decisions.append(decision["decision_point_id"])
        raise AssertionError("runtime decide must not run after Oracle branch-fault abort")


class _AbortingOracle:
    def __init__(self, summary: BranchFaultSummary) -> None:
        self.summary = summary
        self.decisions: list[str] = []

    async def collect(self, instance_id, decision, *, timeout_s):
        del instance_id, timeout_s
        self.decisions.append(decision["decision_point_id"])
        raise BranchFaultAbortError("aborted_snapshot_restore_fault", self.summary)


class OracleFaultAbortRunnerTest(unittest.IsolatedAsyncioTestCase):
    async def test_structural_abort_writes_one_episode_summary_without_runtime_commit(self) -> None:
        summary = BranchFaultSummary(
            search_id="search-structural",
            fault_signature="snapshot_restore_fault",
            fault_kind="worker_exception",
            count=2,
            first_detail=(
                "Snapshot restore rejected: reference_integrity:Dangling reference "
                "'creature-000027'"
            ),
            first_depth=3,
            last_depth=3,
            first_branch_id="branch-a",
            last_branch_id="branch-b",
            action_ids=("end",),
            action_types=("system",),
            root_action_ids=("root-end",),
        )
        client = _Client()
        oracle = _AbortingOracle(summary)
        commit_engine = _CommitEngine(client)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle.jsonl"
            result = await OracleEpisodeRunner(
                client,
                oracle=oracle,  # type: ignore[arg-type]
                commit_engine=commit_engine,  # type: ignore[arg-type]
                writer=OracleJsonlWriter(path),
            ).run(
                "inst",
                oracle_timeout_s=5.0,
                decision_timeout_s=5.0,
            )
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(ORACLE_EPISODE_RESULT_SCHEMA_VERSION, 2)
        self.assertEqual(oracle.decisions, ["d-root"])
        self.assertEqual(commit_engine.decisions, [])
        self.assertEqual(client.commits, [])
        self.assertEqual(client.closed, ["inst"])
        self.assertFalse(result.completed)
        self.assertEqual(result.decisions_collected, 0)
        self.assertEqual(result.termination_reason, "aborted_snapshot_restore_fault")
        self.assertEqual(result.fault_summary, summary)

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["record_type"], "combat_oracle_episode_result")
        self.assertEqual(record["record_schema_version"], 2)
        self.assertFalse(record["completed"])
        self.assertEqual(record["termination_reason"], "aborted_snapshot_restore_fault")
        self.assertEqual(record["fault_summary"]["search_id"], "search-structural")
        self.assertEqual(record["fault_summary"]["fault_signature"], "snapshot_restore_fault")
        self.assertEqual(record["fault_summary"]["count"], 2)
        self.assertEqual(record["fault_summary"]["first_depth"], 3)
        self.assertEqual(record["fault_summary"]["action_types"], ["system"])
        self.assertIn(
            "reference_integrity:",
            record["fault_summary"]["first_detail"],
        )


if __name__ == "__main__":
    unittest.main()
