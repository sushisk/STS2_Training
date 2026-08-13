from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from sts2_training.decision.learned_value import (
    LEARNED_VALUE_ARTIFACT_SCHEMA_VERSION,
    LinearValueModel,
)
from sts2_training.decision.oracle_log import ORACLE_VALUE_MASK_VERSION
from sts2_training.decision.value_features import (
    VALUE_FEATURE_NAMES,
    VALUE_FEATURE_SCHEMA_VERSION,
)


_DTO_VERSION = "emulator-test"


def _payload() -> dict:
    return {
        "model_type": "ridge_linear_combat_value",
        "artifact_schema_version": LEARNED_VALUE_ARTIFACT_SCHEMA_VERSION,
        "feature_schema_version": VALUE_FEATURE_SCHEMA_VERSION,
        "required_mask_version": ORACLE_VALUE_MASK_VERSION,
        "required_dto_version": _DTO_VERSION,
        "feature_names": list(VALUE_FEATURE_NAMES),
        "coefficients": [0.0] * len(VALUE_FEATURE_NAMES),
        "intercept": 7.5,
        "mean": [0.0] * len(VALUE_FEATURE_NAMES),
        "scale": [1.0] * len(VALUE_FEATURE_NAMES),
        "terminal_values": {"victory": 100000.0, "defeat": -100000.0},
    }


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


class LinearValueModelTest(unittest.TestCase):
    def test_artifact_round_trip_and_terminal_override(self) -> None:
        payload = _payload()
        dto = _dto()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.json"
            raw = json.dumps(payload).encode()
            path.write_bytes(raw)
            model = LinearValueModel.from_weights_file(path)

        self.assertEqual(model.evaluate(dto), 7.5)
        self.assertEqual(model.evaluate({**dto, "terminal": True, "outcome": "victory"}), 100000.0)
        provenance = model.oracle_provenance()
        self.assertEqual(provenance["artifact_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(provenance["terminal_values"]["defeat"], -100000.0)
        self.assertEqual(provenance["required_mask_version"], "1.2")
        self.assertEqual(provenance["required_dto_version"], _DTO_VERSION)

    def test_feature_schema_mismatch_fails_closed(self) -> None:
        payload = _payload()
        payload["feature_schema_version"] = 999
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "feature schema mismatch"):
                LinearValueModel.from_weights_file(path)

    def test_artifact_mask_contract_mismatch_fails_closed(self) -> None:
        payload = _payload()
        payload["required_mask_version"] = "1.1"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mask contract mismatch"):
                LinearValueModel.from_weights_file(path)

    def test_runtime_rejects_legacy_mask_even_for_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.json"
            path.write_text(json.dumps(_payload()), encoding="utf-8")
            model = LinearValueModel.from_weights_file(path)

        legacy = {**_dto(), "mask_version": "1.1", "terminal": True, "outcome": "victory"}
        with self.assertRaisesRegex(ValueError, "mask_version='1.2'"):
            model.evaluate(legacy)

    def test_runtime_rejects_different_dto_generation_even_for_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.json"
            path.write_text(json.dumps(_payload()), encoding="utf-8")
            model = LinearValueModel.from_weights_file(path)

        other = {**_dto(), "dto_version": "emulator-other", "terminal": True, "outcome": "victory"}
        with self.assertRaisesRegex(ValueError, "dto_version"):
            model.evaluate(other)


if __name__ == "__main__":
    unittest.main()
