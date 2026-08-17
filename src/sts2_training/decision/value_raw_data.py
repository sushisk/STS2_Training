"""Lossless public Oracle-v7 data loading for future Value-learning consumers.

Structured supervised/RL loaders intentionally validate and normalize the fields they use.
This module provides the complementary foundation seam: keep the complete public JSONL
record, including fields unknown to today's consumer, while still fail-closing on the
known Oracle/DTO contract.  Hidden Emulator state is already excluded upstream by the RL
mask and is never reconstructed here.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sts2_training.decision.oracle_log import (
    ORACLE_EPISODE_RESULT_SCHEMA_VERSION,
    ORACLE_RECORD_SCHEMA_VERSION,
    oracle_value_dto_contract,
)
from sts2_training.decision.value_training_data import (
    OracleValueDtoContract,
    inspect_oracle_value_dto_contract,
)


@dataclass(frozen=True)
class RawOracleValueRecord:
    """One JSONL object with every public producer field preserved verbatim."""

    source_path: str
    line_number: int
    payload: Mapping[str, Any]

    @property
    def record_type(self) -> str | None:
        value = self.payload.get("record_type")
        return value if isinstance(value, str) else None


@dataclass(frozen=True)
class RawCombatValueEpisode:
    """Lossless actual-episode grouping over complete v7/v2 record payloads."""

    instance_id: str
    server_epoch: str | None
    decision_records: tuple[RawOracleValueRecord, ...]
    episode_result: RawOracleValueRecord
    dto_contract: OracleValueDtoContract


def load_oracle_value_raw_records(
    paths: Iterable[str | Path],
) -> tuple[list[RawOracleValueRecord], OracleValueDtoContract]:
    """Load JSONL without projecting away public fields.

    Known `combat_oracle_decision`/`combat_oracle_episode_result` records are validated
    against the current schemas and the exact wire/mask/dto generation. Unknown record
    types are retained unchanged instead of being silently discarded by this foundation
    loader.
    """

    normalized_paths = tuple(Path(path) for path in paths)
    contract = inspect_oracle_value_dto_contract(normalized_paths)
    records: list[RawOracleValueRecord] = []
    for path in normalized_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"{path}:{line_number}: Oracle JSONL record must be an object")
                _validate_known_record(
                    payload,
                    contract=contract,
                    source=f"{path}:{line_number}",
                )
                records.append(
                    RawOracleValueRecord(
                        source_path=str(path),
                        line_number=line_number,
                        payload=copy.deepcopy(dict(payload)),
                    )
                )
    return records, contract


def group_raw_combat_value_episodes(
    records: Iterable[RawOracleValueRecord],
    *,
    dto_contract: OracleValueDtoContract,
) -> list[RawCombatValueEpisode]:
    """Group known actual-trajectory records while retaining each complete payload."""

    decisions: dict[tuple[str | None, str], list[RawOracleValueRecord]] = {}
    results: dict[tuple[str | None, str], RawOracleValueRecord] = {}
    for raw in records:
        payload = raw.payload
        if raw.record_type == "combat_oracle_decision":
            instance_id = _string(payload.get("instance_id"), f"{_source(raw)}.instance_id")
            metadata = _mapping(
                payload.get("decision_response_metadata"),
                f"{_source(raw)}.decision_response_metadata",
            )
            server_epoch = _optional_string(metadata.get("server_epoch"))
            decisions.setdefault((server_epoch, instance_id), []).append(raw)
        elif raw.record_type == "combat_oracle_episode_result":
            instance_id = _string(payload.get("instance_id"), f"{_source(raw)}.instance_id")
            metadata = _mapping(
                payload.get("final_decision_metadata"),
                f"{_source(raw)}.final_decision_metadata",
            )
            server_epoch = _optional_string(metadata.get("server_epoch"))
            key = (server_epoch, instance_id)
            if key in results:
                raise ValueError(f"{_source(raw)}: duplicate episode result for {key!r}")
            results[key] = raw

    episodes: list[RawCombatValueEpisode] = []
    for key in sorted(set(decisions) | set(results), key=lambda item: (str(item[0]), item[1])):
        if key not in results:
            raise ValueError(f"raw actual trajectory {key!r} is missing episode result")
        ordered = sorted(
            decisions.get(key, []),
            key=lambda item: _nonnegative_int(
                item.payload.get("decision_index"),
                f"{_source(item)}.decision_index",
            ),
        )
        for expected, item in enumerate(ordered):
            actual = _nonnegative_int(
                item.payload.get("decision_index"),
                f"{_source(item)}.decision_index",
            )
            if actual != expected:
                raise ValueError(f"raw actual trajectory {key!r} has non-contiguous decisions")
        result = results[key]
        expected_count = _nonnegative_int(
            result.payload.get("decisions_collected"),
            f"{_source(result)}.decisions_collected",
        )
        if expected_count != len(ordered):
            raise ValueError(
                f"raw actual trajectory {key!r} decisions_collected={expected_count} "
                f"but loaded {len(ordered)} decision records"
            )
        episodes.append(
            RawCombatValueEpisode(
                instance_id=key[1],
                server_epoch=key[0],
                decision_records=tuple(ordered),
                episode_result=result,
                dto_contract=dto_contract,
            )
        )
    return episodes


def load_raw_combat_value_episodes(
    paths: Iterable[str | Path],
) -> list[RawCombatValueEpisode]:
    records, contract = load_oracle_value_raw_records(paths)
    return group_raw_combat_value_episodes(records, dto_contract=contract)


def _validate_known_record(
    payload: Mapping[str, Any],
    *,
    contract: OracleValueDtoContract,
    source: str,
) -> None:
    record_type = payload.get("record_type")
    if record_type == "combat_oracle_decision":
        if payload.get("record_schema_version") != ORACLE_RECORD_SCHEMA_VERSION:
            raise ValueError(f"{source}: incompatible Oracle decision schema")
        dto = _mapping(payload.get("masked_emulator_dto"), f"{source}.masked_emulator_dto")
        _validate_dto_contract(payload, dto=dto, contract=contract, source=source)
        # Runtime transition remains required by v7; it was introduced in v6. Keep the
        # full object rather than normalizing it here so diagnostics remain available.
        _mapping(payload.get("runtime_transition"), f"{source}.runtime_transition")
    elif record_type == "combat_oracle_episode_result":
        if payload.get("record_schema_version") != ORACLE_EPISODE_RESULT_SCHEMA_VERSION:
            raise ValueError(f"{source}: incompatible Oracle episode schema")
        dto = _mapping(
            payload.get("final_masked_emulator_dto"),
            f"{source}.final_masked_emulator_dto",
        )
        _validate_dto_contract(payload, dto=dto, contract=contract, source=source)


def _validate_dto_contract(
    payload: Mapping[str, Any],
    *,
    dto: Mapping[str, Any],
    contract: OracleValueDtoContract,
    source: str,
) -> None:
    actual = oracle_value_dto_contract(dto, context=source)
    expected = contract.to_json()
    if actual != expected:
        raise ValueError(
            f"{source}: embedded DTO contract {actual!r} does not match dataset {expected!r}"
        )
    declared = _mapping(payload.get("dto_contract"), f"{source}.dto_contract")
    declared_triplet = {
        "wire_schema_version": declared.get("wire_schema_version"),
        "mask_version": declared.get("mask_version"),
        "dto_version": declared.get("dto_version"),
    }
    if declared_triplet != expected:
        raise ValueError(f"{source}: declared dto_contract does not match dataset contract")


def _source(record: RawOracleValueRecord) -> str:
    return f"{record.source_path}:{record.line_number}"


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


__all__ = [
    "RawCombatValueEpisode",
    "RawOracleValueRecord",
    "group_raw_combat_value_episodes",
    "load_oracle_value_raw_records",
    "load_raw_combat_value_episodes",
]
