from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts2_training.decision.beam_search import BeamNode, BeamSearchResult, BeamSearchStats
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


_DTO_VERSION = "emulator-test"


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
        best_node = BeamNode(
            branch_id="oracle-deep",
            parent_branch_id="oracle-parent",
            rng_id=11,
            decision_point_id="oracle-deep-d",
            masked_emulator_dto={"deep_oracle_payload": "must-not-be-logged"},
            depth=4,
            value=12.5,
            root_action_id="play",
            combat_depth=4,
            branch_log=("deep", "branch", "log"),
            action_id="deep-action",
            action_type="card",
            action={"action_id": "deep-action", "action_type": "card"},
        )
        return OracleCollectionResult(
            search_result=BeamSearchResult(
                best_root_action_id="play",
                best_value=12.5,
                best_node=best_node,
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

    def _dto(self, **extra):
        return {
            "mask_version": ORACLE_VALUE_MASK_VERSION,
            "dto_version": _DTO_VERSION,
            "legal_actions": [],
            **extra,
        }

    def _transition(self, *, next_id: str = "d-next", next_dto=None):
        return {
            "chosen_action_id": "play",
            "chosen_action": {"action_id": "play", "action_type": "card"},
            "next_decision_point_id": next_id,
            "commit_response_metadata": {"decision_point_id": next_id, "status": "completed"},
            "next_masked_emulator_dto": next_dto or self._dto(),
        }

    def test_writer_preserves_public_envelope_dto_oracle_and_actual_transition(self) -> None:
        decision = {
            "status": "completed",
            "server_epoch": "epoch-1",
            "decision_point_id": "d-root",
            "masked_emulator_dto": self._dto(
                hp=37,
                legal_actions=[
                    {"action_id": "play", "action_type": "card", "is_available": True}
                ],
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle.jsonl"
            record = OracleJsonlWriter(path).write(
                decision,
                self._result(),
                instance_id="inst-1",
                decision_index=0,
                runtime_transition=self._transition(),
                training_commit="abc123",
            )
            parsed = json.loads(path.read_text(encoding="utf-8").strip())

        self.assertEqual(record["record_schema_version"], ORACLE_RECORD_SCHEMA_VERSION)
        self.assertEqual(ORACLE_RECORD_SCHEMA_VERSION, 6)
        self.assertEqual(parsed["instance_id"], "inst-1")
        self.assertEqual(parsed["decision_index"], 0)
        self.assertEqual(parsed["decision_response_metadata"]["server_epoch"], "epoch-1")
        self.assertNotIn("masked_emulator_dto", parsed["decision_response_metadata"])
        self.assertEqual(parsed["dto_contract"]["dto_version"], _DTO_VERSION)
        self.assertEqual(parsed["masked_emulator_dto"]["hp"], 37)
        self.assertEqual(parsed["runtime_transition"]["chosen_action_id"], "play")
        self.assertEqual(
            parsed["runtime_transition"]["next_masked_emulator_dto"]["dto_version"],
            _DTO_VERSION,
        )
        self.assertEqual(parsed["oracle_targets"]["metadata"]["oracle_beam_width"], 16)
        oracle_best = parsed["oracle_search_result"]["best_node"]
        self.assertEqual(oracle_best["branch_id"], "oracle-deep")
        self.assertEqual(oracle_best["value"], 12.5)
        self.assertNotIn("masked_emulator_dto", oracle_best)
        self.assertNotIn("branch_log", oracle_best)
        self.assertEqual(
            oracle_best["omitted_large_fields"],
            ["masked_emulator_dto", "branch_log"],
        )
        self.assertEqual(parsed["provenance"]["training_commit"], "abc123")

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
            "masked_emulator_dto": self._dto(
                hand=[rich_card],
                drawPile=[{**rich_card, "count": 2}],
                discardPile=[],
                exhaustPile=[],
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle.jsonl"
            OracleJsonlWriter(path).write(
                decision,
                self._result(),
                instance_id="inst-1",
                decision_index=0,
                runtime_transition=self._transition(),
            )
            parsed = json.loads(path.read_text(encoding="utf-8").strip())

        dto = parsed["masked_emulator_dto"]
        self.assertEqual(dto["hand"][0]["upgradeLevel"], 2)
        self.assertEqual(dto["hand"][0]["enchantment"]["amount"], 3)
        self.assertIsInstance(dto["drawPile"], list)
        self.assertEqual(dto["drawPile"][0]["count"], 2)

    def test_writer_rejects_legacy_mask_or_missing_dto_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = OracleJsonlWriter(Path(tmpdir) / "oracle.jsonl")
            invalid = [
                {"mask_version": "1.1", "dto_version": _DTO_VERSION, "legal_actions": []},
                {"mask_version": "1.2", "legal_actions": []},
            ]
            for dto in invalid:
                with self.subTest(dto=dto), self.assertRaises(ValueError):
                    writer.write(
                        {"decision_point_id": "d-root", "masked_emulator_dto": dto},
                        self._result(),
                        instance_id="inst-1",
                        decision_index=0,
                        runtime_transition=self._transition(),
                    )

    def test_writer_rejects_dto_version_change_across_transition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = OracleJsonlWriter(Path(tmpdir) / "oracle.jsonl")
            bad_next = self._dto(dto_version="emulator-other")
            with self.assertRaisesRegex(ValueError, "dto_version"):
                writer.write(
                    {"decision_point_id": "d-root", "masked_emulator_dto": self._dto()},
                    self._result(),
                    instance_id="inst-1",
                    decision_index=0,
                    runtime_transition=self._transition(next_dto=bad_next),
                )


if __name__ == "__main__":
    unittest.main()
