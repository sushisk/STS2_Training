from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sts2_training.api.contract import SCHEMA_VERSION
from sts2_training.decision.action_score_features import ACTION_SCORE_FEATURE_NAMES
from sts2_training.decision.action_score_training_data import (
    build_pairwise_action_score_examples,
    load_combat_action_score_examples,
)
from sts2_training.decision.oracle_log import ORACLE_RECORD_SCHEMA_VERSION, ORACLE_VALUE_MASK_VERSION

_DTO_VERSION = "emulator-test"


def _dto() -> dict:
    return {
        "mask_version": ORACLE_VALUE_MASK_VERSION,
        "dto_version": _DTO_VERSION,
        "hp": 40,
        "maxHp": 80,
        "block": 0,
        "energy": 3,
        "enemies": [],
        "hand": [],
        "drawPile": [],
        "discardPile": [],
        "exhaustPile": [],
        "potions": [],
        "playerPowers": [],
    }


def _root_action(
    action_id: str,
    estimated_q: float | None,
    *,
    target_source: str,
    rng_values: list[tuple[float | None, str]],
    censored: bool = False,
) -> dict:
    return {
        "action_id": action_id,
        "action": {"action_id": action_id, "action_type": "system", "parameters": {}},
        "evaluated": target_source != "no_target",
        "estimated_q": estimated_q,
        "rng_outcomes": [
            {
                "rng_id": index,
                "value": value,
                "target_source": source,
                "terminal_reached": source == "terminal",
                "deepest_combat_depth": 1,
                "censored": source != "terminal",
                "censor_reason": None if source == "terminal" else "value_bootstrap:depth_limit",
                "best_node_id": None if value is None else f"node-{index}",
            }
            for index, (value, source) in enumerate(rng_values)
        ],
        "target_source": target_source,
        "terminal_reached": target_source == "terminal",
        "censored": censored,
        "censor_reason": "partial" if censored else None,
    }


def _record(*, exhaustive: bool = True) -> dict:
    dto = _dto()
    return {
        "record_type": "combat_oracle_decision",
        "record_schema_version": ORACLE_RECORD_SCHEMA_VERSION,
        "instance_id": "instance-1",
        "decision_index": 0,
        "decision_point_id": "dp-1",
        "dto_contract": {
            "wire_schema_version": SCHEMA_VERSION,
            "mask_version": ORACLE_VALUE_MASK_VERSION,
            "dto_version": _DTO_VERSION,
        },
        "masked_emulator_dto": dto,
        "oracle_targets": {
            "metadata": {"exhaustive_root_actions": exhaustive},
            "root_actions": [
                _root_action("best", 10.0, target_source="terminal", rng_values=[(10.0, "terminal")]),
                _root_action(
                    "bootstrap",
                    5.0,
                    target_source="value_bootstrap",
                    rng_values=[(5.0, "value_bootstrap")],
                    censored=True,
                ),
                _root_action(
                    "partial",
                    7.0,
                    target_source="terminal",
                    rng_values=[(7.0, "terminal"), (None, "no_target")],
                    censored=True,
                ),
                _root_action("none", None, target_source="no_target", rng_values=[]),
            ],
        },
    }


class CombatActionScoreTrainingDataTest(unittest.TestCase):
    def _load(self, record: dict):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with patch(
                "sts2_training.decision.action_score_training_data.inspect_oracle_teacher_provenance"
            ):
                return load_combat_action_score_examples([path])

    def test_loads_resolved_q_targets_and_excludes_unresolved_or_no_target(self) -> None:
        examples, stats = self._load(_record())
        self.assertEqual(ORACLE_RECORD_SCHEMA_VERSION, 7)
        self.assertEqual([example.action_id for example in examples], ["best", "bootstrap"])
        self.assertEqual([example.sample_weight for example in examples], [1.0, 0.5])
        self.assertEqual(stats.root_actions, 4)
        self.assertEqual(stats.usable_actions, 2)
        self.assertEqual(stats.no_target_actions, 1)
        self.assertEqual(stats.unresolved_actions, 1)
        self.assertEqual(stats.dto_version, _DTO_VERSION)

        pairs = build_pairwise_action_score_examples(examples)
        self.assertEqual(len(pairs), 2)
        self.assertEqual({pair.label for pair in pairs}, {0, 1})
        positive = next(pair for pair in pairs if pair.label == 1)
        self.assertEqual(positive.winner_action_id, "best")
        self.assertEqual(positive.loser_action_id, "bootstrap")
        self.assertEqual(positive.sample_weight, 0.5)

    def test_v6_record_is_rejected_instead_of_being_reinterpreted_as_v7(self) -> None:
        legacy = _record()
        legacy["record_schema_version"] = 6
        with self.assertRaisesRegex(ValueError, "expected Oracle record schema v7"):
            self._load(legacy)

    def test_board_context_interaction_survives_pairwise_delta(self) -> None:
        record = _record()
        record["masked_emulator_dto"]["enemies"] = [
            {
                "index": 0,
                "hp": 20,
                "maxHp": 20,
                "block": 0,
                "isAlive": True,
                "intent": {"attackDamage": 40, "attackRepeats": 1},
                "powers": [],
            }
        ]
        record["oracle_targets"]["root_actions"][0]["action"]["action_type"] = "card"

        examples, _ = self._load(record)
        pairs = build_pairwise_action_score_examples(examples)
        positive = next(pair for pair in pairs if pair.label == 1)
        feature_index = ACTION_SCORE_FEATURE_NAMES.index(
            "context_danger_ratio_x_action_card"
        )
        self.assertEqual(positive.feature_delta[feature_index], 1.0)

    def test_non_exhaustive_oracle_is_rejected_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "exhaustive_root_actions=True"):
            self._load(_record(exhaustive=False))


if __name__ == "__main__":
    unittest.main()
