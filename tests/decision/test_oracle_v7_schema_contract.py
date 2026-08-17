from __future__ import annotations

import json
from pathlib import Path

import pytest

from sts2_training.api.contract import SCHEMA_VERSION
from sts2_training.decision.oracle_log import (
    ORACLE_RECORD_SCHEMA_VERSION,
    ORACLE_VALUE_MASK_VERSION,
)
from sts2_training.decision.value_training_data import inspect_oracle_value_dto_contract


def _decision_record(*, schema_version: int) -> dict:
    dto = {
        "mask_version": ORACLE_VALUE_MASK_VERSION,
        "dto_version": "emulator-test",
        "legal_actions": [],
    }
    return {
        "record_type": "combat_oracle_decision",
        "record_schema_version": schema_version,
        "dto_contract": {
            "wire_schema_version": SCHEMA_VERSION,
            "mask_version": ORACLE_VALUE_MASK_VERSION,
            "dto_version": "emulator-test",
        },
        "masked_emulator_dto": dto,
    }


def _write(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_value_data_contract_accepts_oracle_v7(tmp_path: Path) -> None:
    assert ORACLE_RECORD_SCHEMA_VERSION == 7
    path = tmp_path / "oracle-v7.jsonl"
    _write(path, _decision_record(schema_version=7))

    contract = inspect_oracle_value_dto_contract([path])

    assert contract.wire_schema_version == SCHEMA_VERSION
    assert contract.mask_version == ORACLE_VALUE_MASK_VERSION
    assert contract.dto_version == "emulator-test"


def test_value_data_contract_rejects_oracle_v6(tmp_path: Path) -> None:
    path = tmp_path / "oracle-v6.jsonl"
    _write(path, _decision_record(schema_version=6))

    with pytest.raises(ValueError, match="expected Oracle decision schema v7"):
        inspect_oracle_value_dto_contract([path])
