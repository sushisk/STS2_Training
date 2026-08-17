from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts2_training.api.contract import SCHEMA_VERSION
from sts2_training.decision.oracle_log import (
    ORACLE_EPISODE_RESULT_SCHEMA_VERSION,
    ORACLE_RECORD_SCHEMA_VERSION,
)
from sts2_training.decision.value_raw_data import (
    load_oracle_value_raw_records,
    load_raw_combat_value_episodes,
)


_DTO_VERSION = "emulator-test"


def _contract() -> dict:
    return {
        "wire_schema_version": SCHEMA_VERSION,
        "mask_version": "1.2",
        "dto_version": _DTO_VERSION,
    }


def _dto(*, terminal: bool = False) -> dict:
    dto = {
        "mask_version": "1.2",
        "dto_version": _DTO_VERSION,
        "hp": 50,
        "maxHp": 80,
        "legal_actions": [],
    }
    if terminal:
        dto.update({"terminal": True, "outcome": "victory"})
    return dto


def _decision_record() -> dict:
    return {
        "record_type": "combat_oracle_decision",
        "record_schema_version": ORACLE_RECORD_SCHEMA_VERSION,
        "instance_id": "inst-1",
        "decision_index": 0,
        "decision_point_id": "d0",
        "dto_contract": _contract(),
        "decision_response_metadata": {"server_epoch": "epoch-1", "future_envelope": 17},
        "masked_emulator_dto": _dto(),
        "root_value_samples": [],
        "runtime_transition": {
            "chosen_action_id": "a",
            "chosen_action": {"action_id": "a", "action_type": "card"},
            "decision_source": "beam_search",
            "beam_result": {"best_root_action_id": "a", "best_value": 2.5},
            "next_decision_point_id": "d1",
            "commit_response_metadata": {"server_epoch": "epoch-1"},
            "next_masked_emulator_dto": _dto(terminal=True),
            "future_runtime_diagnostic": {"latency_ms": 3.5},
        },
        "future_decision_field": {"producer": "kept"},
    }


def _episode_record() -> dict:
    return {
        "record_type": "combat_oracle_episode_result",
        "record_schema_version": ORACLE_EPISODE_RESULT_SCHEMA_VERSION,
        "instance_id": "inst-1",
        "decisions_collected": 1,
        "completed": True,
        "termination_reason": "terminal",
        "combat_result": "victory",
        "dto_contract": _contract(),
        "final_decision_metadata": {"server_epoch": "epoch-1"},
        "final_masked_emulator_dto": _dto(terminal=True),
        "future_episode_field": [1, 2, 3],
    }


class ValueRawDataTest(unittest.TestCase):
    def test_raw_loader_preserves_known_and_future_public_fields(self) -> None:
        decision = _decision_record()
        episode = _episode_record()
        unknown = {"record_type": "future_oracle_record", "public_payload": {"x": 1}}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle.jsonl"
            path.write_text(
                "\n".join(json.dumps(item) for item in (decision, unknown, episode)) + "\n",
                encoding="utf-8",
            )
            records, contract = load_oracle_value_raw_records([path])

        self.assertEqual(ORACLE_RECORD_SCHEMA_VERSION, 7)
        self.assertEqual(records[0].payload["record_schema_version"], 7)
        self.assertEqual(contract.dto_version, _DTO_VERSION)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0].payload["future_decision_field"]["producer"], "kept")
        transition = records[0].payload["runtime_transition"]
        self.assertEqual(transition["decision_source"], "beam_search")
        self.assertEqual(transition["beam_result"]["best_value"], 2.5)
        self.assertEqual(transition["future_runtime_diagnostic"]["latency_ms"], 3.5)
        self.assertEqual(records[1].payload["public_payload"]["x"], 1)
        self.assertEqual(records[2].payload["future_episode_field"], [1, 2, 3])

        # The loader owns a deep copy rather than sharing mutable producer fixtures.
        decision["future_decision_field"]["producer"] = "mutated"
        self.assertEqual(records[0].payload["future_decision_field"]["producer"], "kept")

    def test_raw_loader_rejects_v6_decision_records(self) -> None:
        legacy = _decision_record()
        legacy["record_schema_version"] = 6
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle-v6.jsonl"
            path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incompatible Oracle decision schema"):
                load_oracle_value_raw_records([path])

    def test_raw_episode_group_keeps_complete_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle.jsonl"
            path.write_text(
                json.dumps(_decision_record()) + "\n" + json.dumps(_episode_record()) + "\n",
                encoding="utf-8",
            )
            episodes = load_raw_combat_value_episodes([path])

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].instance_id, "inst-1")
        self.assertEqual(episodes[0].server_epoch, "epoch-1")
        self.assertEqual(len(episodes[0].decision_records), 1)
        self.assertEqual(
            episodes[0].decision_records[0].payload["runtime_transition"]["decision_source"],
            "beam_search",
        )
        self.assertEqual(episodes[0].episode_result.payload["future_episode_field"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
