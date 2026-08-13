from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts2_training.decision.beam_search import BeamSearchResult, BeamSearchStats
from sts2_training.decision.oracle_log import (
    ORACLE_RECORD_SCHEMA_VERSION,
    ORACLE_VALUE_MASK_VERSION,
    OracleJsonlWriter,
)
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
                "mask_version": ORACLE_VALUE_MASK_VERSION,
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
        self.assertEqual(parsed["masked_emulator_dto"]["mask_version"], "1.2")
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

    def test_writer_preserves_upgrade_and_enchantment_card_identity(self) -> None:
        rich_card = {
            "id": "STRIKE_IRONCLAD",
            "type": "Attack",
            "rarity": "Basic",
            "cost": 1,
            "targetType": "AnyEnemy",
            "upgraded": True,
            "upgradeLevel": 2,
            "tinkerTimeType": None,
            "tinkerTimeRider": None,
            "enchantment": {"id": "SHARP", "amount": 3, "status": "Normal"},
        }
        decision = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": {
                "mask_version": ORACLE_VALUE_MASK_VERSION,
                "hand": [rich_card],
                "drawPile": [{**rich_card, "count": 2}],
                "discardPile": [],
                "exhaustPile": [],
                "legal_actions": [],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle.jsonl"
            OracleJsonlWriter(path).write(decision, self._result())
            parsed = json.loads(path.read_text(encoding="utf-8").strip())

        dto = parsed["masked_emulator_dto"]
        self.assertEqual(dto["hand"][0]["upgradeLevel"], 2)
        self.assertEqual(dto["hand"][0]["enchantment"]["amount"], 3)
        self.assertIsInstance(dto["drawPile"], list)
        self.assertEqual(dto["drawPile"][0]["count"], 2)
        self.assertEqual(dto["drawPile"][0]["enchantment"]["id"], "SHARP")

    def test_writer_rejects_legacy_or_missing_mask_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle.jsonl"
            writer = OracleJsonlWriter(path)
            for version in (None, "1.1"):
                dto = {"legal_actions": []}
                if version is not None:
                    dto["mask_version"] = version
                with self.subTest(version=version), self.assertRaisesRegex(
                    ValueError, "mask_version='1.2'"
                ):
                    writer.write(
                        {
                            "decision_point_id": "d-root",
                            "masked_emulator_dto": dto,
                        },
                        self._result(),
                    )

    def test_writer_appends_one_record_per_decision(self) -> None:
        decision = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": {
                "mask_version": ORACLE_VALUE_MASK_VERSION,
                "legal_actions": [],
            },
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
