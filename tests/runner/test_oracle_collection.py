from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sts2_training.decision.beam_search import BeamSearchResult, BeamSearchStats
from sts2_training.decision.oracle_log import ORACLE_VALUE_MASK_VERSION, OracleJsonlWriter
from sts2_training.decision.oracle_search import (
    OracleCollectionResult,
    OracleProvenance,
    OracleTargetMetadata,
    OracleTargets,
)
from sts2_training.runner.oracle_collection import OracleEpisodeRunner, _parse_args


class _FakeClient:
    def __init__(self) -> None:
        self.closed: list[str] = []
        self.commits: list[str] = []

    async def get_decision(self, instance_id, branch_id, *, timeout_s):
        return {
            "decision_point_id": "d-root",
            "masked_emulator_dto": {
                "mask_version": ORACLE_VALUE_MASK_VERSION,
                "hp": 30,
                "legal_actions": [
                    {"action_id": "a", "action_type": "card", "is_available": True},
                    {"action_id": "end", "action_type": "system", "is_available": True},
                ],
            },
        }

    async def commit_action(self, instance_id, decision_point_id, action_id, *, timeout_s):
        self.commits.append(action_id)
        return {
            "decision_point_id": "d-terminal",
            "masked_emulator_dto": {
                "mask_version": ORACLE_VALUE_MASK_VERSION,
                "terminal": True,
                "outcome": "victory",
                "hand": [
                    {
                        "id": "STRIKE_IRONCLAD",
                        "type": "Attack",
                        "upgradeLevel": 2,
                        "enchantment": {"id": "SHARP", "amount": 3, "status": "Normal"},
                    }
                ],
                "drawPile": [
                    {
                        "id": "DEFEND_IRONCLAD",
                        "type": "Skill",
                        "upgradeLevel": 1,
                        "enchantment": None,
                        "count": 2,
                    }
                ],
                "legal_actions": [],
            },
        }

    async def close_instance(self, instance_id, *, timeout_s):
        self.closed.append(instance_id)


class _FakeCommitEngine:
    def __init__(self, client) -> None:
        self.client = client
        self.beam_search = SimpleNamespace()
        self.decisions: list[str] = []

    async def decide(self, instance_id, *, timeout_s, decision):
        self.decisions.append(decision["decision_point_id"])
        return SimpleNamespace(chosen_action_id="a")


class _FailingCommitEngine(_FakeCommitEngine):
    async def decide(self, instance_id, *, timeout_s, decision):
        self.decisions.append(decision["decision_point_id"])
        raise RuntimeError("commit decision failed")


class _FakeOracle:
    def __init__(self) -> None:
        self.decisions: list[str] = []

    async def collect(self, instance_id, decision, *, timeout_s):
        self.decisions.append(decision["decision_point_id"])
        metadata = OracleTargetMetadata(
            search_id="search",
            oracle_beam_width=8,
            target_beam_width=2,
            top_k_actions=4,
            max_depth=3,
            max_continuation_steps=8,
            time_budget_ms=None,
            exhaustive_root_actions=True,
            rng_sampling="independent",
            search_reason="max_depth",
            pruner_name="value_top_k",
            pruner_version="1",
        )
        return OracleCollectionResult(
            search_result=BeamSearchResult(
                best_root_action_id="a",
                best_value=1.0,
                best_node=None,
                reason="max_depth",
                stats=BeamSearchStats(),
            ),
            trace=(),
            targets=OracleTargets(metadata=metadata, root_actions=(), stable_nodes=()),
            provenance=OracleProvenance(
                teacher_policy_class="teacher.Coverage",
                teacher_inner_policy_class="teacher.Policy",
                teacher_coverage_policy_class="teacher.Coverage",
                teacher_value_class="teacher.Value",
            ),
        )


class OracleEpisodeRunnerTest(unittest.IsolatedAsyncioTestCase):
    async def test_collects_before_runtime_commit_and_appends_combat_result(self) -> None:
        client = _FakeClient()
        oracle = _FakeOracle()
        commit_engine = _FakeCommitEngine(client)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle.jsonl"
            runner = OracleEpisodeRunner(
                client,
                oracle=oracle,  # type: ignore[arg-type]
                commit_engine=commit_engine,  # type: ignore[arg-type]
                writer=OracleJsonlWriter(path),
                training_commit="abc",
            )
            result = await runner.run(
                "inst",
                oracle_timeout_s=5.0,
                decision_timeout_s=5.0,
            )
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(oracle.decisions, ["d-root"])
        self.assertEqual(commit_engine.decisions, ["d-root"])
        self.assertEqual(client.commits, ["a"])
        self.assertEqual(client.closed, ["inst"])
        self.assertEqual(result.decisions_collected, 1)
        self.assertEqual(result.final_dto["outcome"], "victory")
        self.assertTrue(result.completed)
        self.assertEqual(result.combat_result, "victory")
        self.assertEqual(result.termination_reason, "terminal")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["record_type"], "combat_oracle_decision")
        self.assertEqual(records[0]["decision_point_id"], "d-root")
        self.assertEqual(records[0]["root_value_samples"], [])
        self.assertEqual(records[0]["provenance"]["training_commit"], "abc")
        self.assertEqual(records[0]["provenance"]["teacher_inner_policy_class"], "teacher.Policy")
        self.assertEqual(records[1]["record_type"], "combat_oracle_episode_result")
        self.assertTrue(records[1]["completed"])
        self.assertEqual(records[1]["termination_reason"], "terminal")
        self.assertEqual(records[1]["combat_result"], "victory")
        final = records[1]["final_masked_emulator_dto"]
        self.assertEqual(final["outcome"], "victory")
        self.assertEqual(final["mask_version"], "1.2")
        self.assertEqual(final["hand"][0]["upgradeLevel"], 2)
        self.assertEqual(final["hand"][0]["enchantment"]["amount"], 3)
        self.assertIsInstance(final["drawPile"], list)
        self.assertEqual(final["drawPile"][0]["count"], 2)

    async def test_commit_failure_rolls_back_partial_episode_output(self) -> None:
        client = _FakeClient()
        oracle = _FakeOracle()
        commit_engine = _FailingCommitEngine(client)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle.jsonl"
            prefix = '{"record_type":"previous_complete_episode"}\n'
            path.write_text(prefix, encoding="utf-8")
            runner = OracleEpisodeRunner(
                client,
                oracle=oracle,  # type: ignore[arg-type]
                commit_engine=commit_engine,  # type: ignore[arg-type]
                writer=OracleJsonlWriter(path),
            )

            with self.assertRaisesRegex(RuntimeError, "commit decision failed"):
                await runner.run(
                    "inst",
                    oracle_timeout_s=5.0,
                    decision_timeout_s=5.0,
                )

            output = path.read_text(encoding="utf-8")

        self.assertEqual(oracle.decisions, ["d-root"])
        self.assertEqual(commit_engine.decisions, ["d-root"])
        self.assertEqual(client.closed, ["inst"])
        self.assertEqual(output, prefix)

    def test_cli_defaults_to_exhaustive_root_and_runtime_target_beam(self) -> None:
        args = _parse_args(["--scenario", "scenario.json", "--output", "oracle.jsonl"])

        self.assertFalse(args.policy_limited_root)
        self.assertEqual(args.oracle_beam_width, 32)
        self.assertEqual(args.oracle_depth, 4)
        self.assertIsNone(args.target_beam_width)


if __name__ == "__main__":
    unittest.main()
