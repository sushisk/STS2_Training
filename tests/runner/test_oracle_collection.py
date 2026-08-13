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


_DTO_VERSION = "emulator-test"


class _FakeClient:
    def __init__(self) -> None:
        self.closed: list[str] = []
        self.commits: list[str] = []

    async def get_decision(self, instance_id, branch_id, *, timeout_s):
        return {
            "status": "completed",
            "server_epoch": "epoch-1",
            "decision_point_id": "d-root",
            "masked_emulator_dto": {
                "mask_version": ORACLE_VALUE_MASK_VERSION,
                "dto_version": _DTO_VERSION,
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
            "status": "completed",
            "server_epoch": "epoch-1",
            "decision_point_id": "d-terminal",
            "masked_emulator_dto": {
                "mask_version": ORACLE_VALUE_MASK_VERSION,
                "dto_version": _DTO_VERSION,
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
        return SimpleNamespace(
            chosen_action_id="a",
            source="beam_search",
            beam_result=BeamSearchResult(
                best_root_action_id="a",
                best_value=2.5,
                best_node=None,
                reason="max_depth",
                stats=BeamSearchStats(depths_completed=2, nodes_expanded=4),
            ),
        )


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
    async def test_collects_teacher_then_logs_actual_runtime_transition_and_result(self) -> None:
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
        self.assertTrue(result.completed)
        self.assertEqual(result.combat_result, "victory")
        self.assertEqual(len(records), 2)

        decision_record = records[0]
        self.assertEqual(decision_record["record_type"], "combat_oracle_decision")
        self.assertEqual(decision_record["instance_id"], "inst")
        self.assertEqual(decision_record["decision_index"], 0)
        self.assertEqual(decision_record["decision_point_id"], "d-root")
        self.assertEqual(decision_record["dto_contract"]["dto_version"], _DTO_VERSION)
        self.assertEqual(decision_record["decision_response_metadata"]["server_epoch"], "epoch-1")
        self.assertEqual(decision_record["root_value_samples"], [])
        self.assertEqual(decision_record["provenance"]["training_commit"], "abc")
        transition = decision_record["runtime_transition"]
        self.assertEqual(transition["chosen_action_id"], "a")
        self.assertEqual(transition["chosen_action"]["action_type"], "card")
        self.assertEqual(transition["decision_source"], "beam_search")
        self.assertEqual(transition["beam_result"]["best_root_action_id"], "a")
        self.assertEqual(transition["beam_result"]["best_value"], 2.5)
        self.assertEqual(transition["next_decision_point_id"], "d-terminal")
        self.assertEqual(transition["combat_result"], "victory")
        self.assertEqual(transition["next_dto_contract"]["dto_version"], _DTO_VERSION)
        self.assertEqual(transition["next_masked_emulator_dto"]["outcome"], "victory")

        episode_record = records[1]
        self.assertEqual(episode_record["record_type"], "combat_oracle_episode_result")
        self.assertTrue(episode_record["completed"])
        self.assertEqual(episode_record["combat_result"], "victory")
        self.assertEqual(episode_record["dto_contract"]["dto_version"], _DTO_VERSION)
        self.assertEqual(episode_record["final_decision_metadata"]["server_epoch"], "epoch-1")
        final = episode_record["final_masked_emulator_dto"]
        self.assertEqual(final["hand"][0]["upgradeLevel"], 2)
        self.assertEqual(final["hand"][0]["enchantment"]["amount"], 3)
        self.assertEqual(final["drawPile"][0]["count"], 2)

    async def test_commit_failure_rolls_back_episode_output(self) -> None:
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

        self.assertEqual(output, prefix)
        self.assertEqual(client.closed, ["inst"])

    def test_cli_defaults_to_exhaustive_root_and_runtime_target_beam(self) -> None:
        args = _parse_args(["--scenario", "scenario.json", "--output", "oracle.jsonl"])
        self.assertFalse(args.policy_limited_root)
        self.assertEqual(args.oracle_beam_width, 32)
        self.assertEqual(args.oracle_depth, 4)
        self.assertIsNone(args.target_beam_width)


if __name__ == "__main__":
    unittest.main()
