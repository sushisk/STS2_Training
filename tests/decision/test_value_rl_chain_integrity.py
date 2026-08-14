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
from sts2_training.decision.value_training_data import load_combat_value_rl_episodes


_DTO_VERSION = "emulator-test"


def _contract() -> dict:
    return {
        "wire_schema_version": SCHEMA_VERSION,
        "mask_version": "1.2",
        "dto_version": _DTO_VERSION,
    }


def _dto(*, hp: int = 40) -> dict:
    return {
        "mask_version": "1.2",
        "dto_version": _DTO_VERSION,
        "hp": hp,
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
        "legal_actions": [],
    }


def _decision(index: int, *, before: dict, after: dict) -> dict:
    return {
        "record_type": "combat_oracle_decision",
        "record_schema_version": ORACLE_RECORD_SCHEMA_VERSION,
        "instance_id": "inst-1",
        "decision_index": index,
        "decision_point_id": f"d{index}",
        "dto_contract": _contract(),
        "decision_response_metadata": {
            "decision_point_id": f"d{index}",
            "server_epoch": "epoch-1",
        },
        "masked_emulator_dto": before,
        "runtime_transition": {
            "chosen_action_id": "actual",
            "chosen_action": {"action_id": "actual", "action_type": "system"},
            "next_decision_point_id": f"d{index + 1}",
            "commit_response_metadata": {
                "decision_point_id": f"d{index + 1}",
                "server_epoch": "epoch-1",
            },
            "next_masked_emulator_dto": after,
            "next_dto_contract": _contract(),
            "combat_result": None,
        },
    }


def _episode(final_dto: dict) -> dict:
    return {
        "record_type": "combat_oracle_episode_result",
        "record_schema_version": ORACLE_EPISODE_RESULT_SCHEMA_VERSION,
        "instance_id": "inst-1",
        "decisions_collected": 2,
        "completed": False,
        "termination_reason": "max_decisions",
        "combat_result": None,
        "dto_contract": _contract(),
        "final_decision_metadata": {
            "decision_point_id": "d2",
            "server_epoch": "epoch-1",
        },
        "final_masked_emulator_dto": final_dto,
        "elapsed_s": 1.0,
    }


def _load(*records: dict):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "oracle.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        return load_combat_value_rl_episodes([path])


class CombatValueRLChainIntegrityTest(unittest.TestCase):
    def test_accepts_contiguous_transition_chain(self) -> None:
        start = _dto(hp=40)
        middle = _dto(hp=35)
        final = _dto(hp=30)
        episodes = _load(
            _decision(0, before=start, after=middle),
            _decision(1, before=middle, after=final),
            _episode(final),
        )
        self.assertEqual(len(episodes), 1)
        self.assertEqual(len(episodes[0].steps), 2)

    def test_rejects_broken_decision_point_chain(self) -> None:
        start = _dto(hp=40)
        middle = _dto(hp=35)
        final = _dto(hp=30)
        first = _decision(0, before=start, after=middle)
        first["runtime_transition"]["next_decision_point_id"] = "other"
        with self.assertRaisesRegex(ValueError, "broken decision_point_id chain"):
            _load(first, _decision(1, before=middle, after=final), _episode(final))

    def test_rejects_broken_dto_chain(self) -> None:
        start = _dto(hp=40)
        logged_middle = _dto(hp=35)
        next_step_middle = _dto(hp=34)
        final = _dto(hp=30)
        with self.assertRaisesRegex(ValueError, "broken masked_emulator_dto chain"):
            _load(
                _decision(0, before=start, after=logged_middle),
                _decision(1, before=next_step_middle, after=final),
                _episode(final),
            )


if __name__ == "__main__":
    unittest.main()
