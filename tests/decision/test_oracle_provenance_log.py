from __future__ import annotations

import unittest

from sts2_training.decision.beam_search import BeamSearchResult, BeamSearchStats
from sts2_training.decision.oracle_log import (
    ORACLE_RECORD_SCHEMA_VERSION,
    ORACLE_VALUE_MASK_VERSION,
    oracle_collection_record,
)
from sts2_training.decision.oracle_search import (
    OracleCollectionResult,
    OracleProvenance,
    OracleTargetMetadata,
    OracleTargets,
)


class OracleProvenanceLogTest(unittest.TestCase):
    def test_jsonl_record_persists_teacher_configuration_metadata(self) -> None:
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
        result = OracleCollectionResult(
            search_result=BeamSearchResult(
                best_root_action_id=None,
                best_value=None,
                best_node=None,
                reason="max_depth",
                stats=BeamSearchStats(),
            ),
            trace=(),
            targets=OracleTargets(metadata=metadata, root_actions=(), stable_nodes=()),
            provenance=OracleProvenance(
                teacher_policy_class="example.Policy",
                teacher_inner_policy_class="example.InnerPolicy",
                teacher_coverage_policy_class="example.CoveragePolicy",
                teacher_value_class="example.Value",
                teacher_policy_metadata={"wrapper": 1},
                teacher_inner_policy_metadata={"checkpoint": "policy-v2"},
                teacher_value_metadata={"checkpoint": "value-v7", "config_hash": "abc"},
            ),
        )

        record = oracle_collection_record(
            {
                "decision_point_id": "d-root",
                "masked_emulator_dto": {
                    "mask_version": ORACLE_VALUE_MASK_VERSION,
                    "legal_actions": [],
                },
            },
            result,
        )

        self.assertEqual(ORACLE_RECORD_SCHEMA_VERSION, 5)
        self.assertEqual(record["record_schema_version"], 5)
        self.assertEqual(record["root_value_samples"], [])
        self.assertEqual(
            record["provenance"]["teacher_value_metadata"]["checkpoint"],
            "value-v7",
        )
        self.assertEqual(
            record["provenance"]["teacher_inner_policy_metadata"]["checkpoint"],
            "policy-v2",
        )


if __name__ == "__main__":
    unittest.main()
