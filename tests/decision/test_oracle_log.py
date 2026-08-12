from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts2_training.decision.beam_search import BeamSearchResult, BeamSearchStats
from sts2_training.decision.oracle_log import ORACLE_RECORD_SCHEMA_VERSION, OracleJsonlWriter
from sts2_training.decision.oracle_search import (
    OracleCollectionResult,
    OracleProvenance,
    OracleTargetMetadata,
    OracleTargets,
)


class OracleJsonlWriterTest(unittest.TestCase):
    def _result(self) -> OracleCollectionResult:
        metadata = OracleTargetMetadata(
            search_id="search-1",
            oracle_beam_width=16,
            target_beam_width=4,
            top_k_actions=8,
            max_depth=4,
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
                best_root_action_id="play",
                best_value=12.5,
                best_node=None,
                reason="max_depth",
                stats=BeamSearchStats(depths_completed=4, nodes_expanded=10),
            ),
            trace=(),
            targets=OracleTargets(metadata=metadata, root_actions=(), stable_nodes=()),
            provenance=OracleProvenance(
                teacher_policy_class="example.CoveragePolicy",
                teacher_inner_policy_class="example.Policy",
                teacher_coverage_policy_class="example.CoveragePolicy",
                teacher_value_class="example.Value",
            ),
        )

    def test_writer_preserves_masked_dto_targets_and_provenance(self) -> None:
        decision = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": {
                "hp": 37,
                "legal_actions": [
                    {"action_id": "play", "action_type": "card", "is_available": True}
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle.jsonl"
            record = OracleJsonlWriter(path).write(
                decision,
                self._result(),
                training_commit="abc123",
            )
            parsed = json.loads(path.read_text(encoding="utf-8").strip())

        self.assertEqual(record["record_schema_version"], ORACLE_RECORD_SCHEMA_VERSION)
        self.assertEqual(parsed["decision_point_id"], "d-root")
        self.assertEqual(parsed["masked_emulator_dto"]["hp"], 37)
        self.assertEqual(parsed["oracle_targets"]["metadata"]["oracle_beam_width"], 16)
        self.assertEqual(parsed["provenance"]["training_commit"], "abc123")
        self.assertEqual(parsed["provenance"]["rng_sampling"], "independent")
        self.assertEqual(parsed["provenance"]["teacher_policy_class"], "example.CoveragePolicy")
        self.assertEqual(parsed["provenance"]["teacher_inner_policy_class"], "example.Policy")
        self.assertEqual(
            parsed["provenance"]["teacher_coverage_policy_class"],
            "example.CoveragePolicy",
        )
        self.assertEqual(parsed["provenance"]["teacher_value_class"], "example.Value")

    def test_writer_preserves_card_upgrade_level_and_enchantment(self) -> None:
        """oracle_collection_record() keeps the raw masked_emulator_dto verbatim so
        future feature extractors can be rebuilt without replaying the emulator (see
        its docstring) - confirm that guarantee actually covers the per-card
        upgradeLevel/enchantment fields (STS2_Emulator#7 / STS2_RL#41), not just
        scalar top-level fields like hp."""
        decision = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": {
                "hand": [
                    {
                        "id": "WITHER",
                        "type": "Status",
                        "upgraded": True,
                        "upgradeLevel": 2,
                        "enchantment": None,
                    },
                    {
                        "id": "STRIKE_IRONCLAD",
                        "type": "Attack",
                        "upgraded": False,
                        "upgradeLevel": 0,
                        "enchantment": {"id": "SHARP", "amount": 3, "status": "Normal"},
                    },
                ],
                "legal_actions": [],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle.jsonl"
            OracleJsonlWriter(path).write(decision, self._result(), training_commit="abc123")
            parsed = json.loads(path.read_text(encoding="utf-8").strip())

        hand = parsed["masked_emulator_dto"]["hand"]
        wither = next(c for c in hand if c["id"] == "WITHER")
        self.assertTrue(wither["upgraded"])
        self.assertEqual(wither["upgradeLevel"], 2)

        strike = next(c for c in hand if c["id"] == "STRIKE_IRONCLAD")
        self.assertEqual(strike["enchantment"], {"id": "SHARP", "amount": 3, "status": "Normal"})

    def test_writer_appends_one_record_per_decision(self) -> None:
        decision = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": {"legal_actions": []},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle.jsonl"
            writer = OracleJsonlWriter(path)
            for index in range(2):
                writer.write(
                    {**decision, "decision_point_id": f"d-{index}"},
                    self._result(),
                )
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[1])["decision_point_id"], "d-1")


if __name__ == "__main__":
    unittest.main()