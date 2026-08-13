from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

from sts2_training.decision.beam_search import BeamSearchResult, BeamSearchStats
from sts2_training.decision.oracle_log import (
    ORACLE_VALUE_MASK_VERSION,
    oracle_collection_record,
)
from sts2_training.decision.oracle_search import (
    OracleProvenance,
    OracleTargetMetadata,
    OracleTargets,
)


_DTO_VERSION = "emulator-test"


def _metadata() -> OracleTargetMetadata:
    return OracleTargetMetadata(
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


def _rich_dto() -> dict:
    card = {
        "id": "STRIKE_IRONCLAD",
        "type": "Attack",
        "rarity": "Basic",
        "cost": 1,
        "targetType": "AnyEnemy",
        "upgraded": True,
        "upgradeLevel": 2,
        "tinkerTimeType": "Alpha",
        "tinkerTimeRider": "Beta",
        "enchantment": {"id": "SHARP", "amount": 3, "status": "Normal"},
    }
    return {
        "mask_version": ORACLE_VALUE_MASK_VERSION,
        "dto_version": _DTO_VERSION,
        "hp": 42,
        "hand": [card],
        "drawPile": [{**card, "count": 2}],
        "discardPile": [],
        "exhaustPile": [],
        "legal_actions": [],
    }


def _result(sample_dto: dict):
    return SimpleNamespace(
        search_result=BeamSearchResult(
            best_root_action_id="a",
            best_value=9.0,
            best_node=None,
            reason="max_depth",
            stats=BeamSearchStats(),
        ),
        trace=(),
        targets=OracleTargets(metadata=_metadata(), root_actions=(), stable_nodes=()),
        provenance=OracleProvenance(
            teacher_policy_class="teacher.Policy",
            teacher_inner_policy_class="teacher.Policy",
            teacher_coverage_policy_class=None,
            teacher_value_class="teacher.Value",
        ),
        root_value_samples=(
            {
                "action_id": "a",
                "action": {"action_id": "a", "action_type": "card"},
                "rng_id": 7,
                "root_state_node_id": "search:root-a",
                "decision_point_id": "after-a",
                "masked_emulator_dto": sample_dto,
                "target_value": 9.0,
                "target_source": "value_bootstrap",
                "terminal_reached": False,
                "deepest_combat_depth": 2,
                "censored": True,
                "censor_reason": "value_bootstrap:max_depth",
                "best_node_id": "search:leaf",
            },
        ),
    )


def _transition(dto: dict) -> dict:
    return {
        "chosen_action_id": "a",
        "chosen_action": {"action_id": "a", "action_type": "card"},
        "next_decision_point_id": "d-next",
        "commit_response_metadata": {"decision_point_id": "d-next"},
        "next_masked_emulator_dto": dto,
    }


class OracleValueMaskV12Test(unittest.TestCase):
    def test_root_value_sample_preserves_full_card_identity_and_context(self) -> None:
        dto = _rich_dto()
        record = oracle_collection_record(
            {"decision_point_id": "d-root", "masked_emulator_dto": _rich_dto()},
            _result(dto),
            instance_id="inst-1",
            decision_index=0,
            runtime_transition=_transition(_rich_dto()),
        )

        sample = record["root_value_samples"][0]
        saved = sample["masked_emulator_dto"]
        self.assertEqual(saved, dto)
        self.assertIsNot(saved, dto)
        self.assertEqual(sample["action"]["action_type"], "card")
        self.assertEqual(sample["deepest_combat_depth"], 2)
        self.assertEqual(saved["hand"][0]["upgradeLevel"], 2)
        self.assertEqual(saved["hand"][0]["enchantment"]["amount"], 3)
        self.assertEqual(saved["hand"][0]["tinkerTimeType"], "Alpha")
        self.assertIsInstance(saved["drawPile"], list)
        self.assertEqual(saved["drawPile"][0]["count"], 2)

    def test_root_value_sample_rejects_legacy_mask_independently_of_root_decision(self) -> None:
        legacy = copy.deepcopy(_rich_dto())
        legacy["mask_version"] = "1.1"
        with self.assertRaisesRegex(ValueError, "root_value_samples.*mask_version='1.2'"):
            oracle_collection_record(
                {"decision_point_id": "d-root", "masked_emulator_dto": _rich_dto()},
                _result(legacy),
                instance_id="inst-1",
                decision_index=0,
                runtime_transition=_transition(_rich_dto()),
            )

    def test_root_value_sample_rejects_different_dto_generation(self) -> None:
        mixed = copy.deepcopy(_rich_dto())
        mixed["dto_version"] = "emulator-other"
        with self.assertRaisesRegex(ValueError, "dto_version"):
            oracle_collection_record(
                {"decision_point_id": "d-root", "masked_emulator_dto": _rich_dto()},
                _result(mixed),
                instance_id="inst-1",
                decision_index=0,
                runtime_transition=_transition(_rich_dto()),
            )


if __name__ == "__main__":
    unittest.main()
