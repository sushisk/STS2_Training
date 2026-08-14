"""Harvest `CombatScenario` JSON specs from collected self-play JSONL logs.

The v0.8 wire contract is a hard cutover paired with masked_emulator_dto mask_version
1.2. Card state is preserved for hand and masked pile multisets; pile order alone is
intentionally unavailable.

Harvested scenario JSON remains a pure ``CombatScenario`` input. Dataset lineage is
written separately to ``harvest_manifest.json`` so provenance cannot leak into the
Emulator wire shape. Train/validation/test assignment is deterministic by source-run
SHA-256, keeping sibling combats from one Whole Run in the same split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]
_GOD_MODE_POWER_IDS = frozenset({"STRENGTH_POWER", "BUFFER_POWER", "REGEN_POWER"})
_MASK_VERSION = "1.2"
HARVEST_MANIFEST_SCHEMA_VERSION = 1

__all__ = [
    "HARVEST_MANIFEST_SCHEMA_VERSION",
    "dto_to_scenario_spec",
    "harvest_scenario_records_from_jsonl",
    "harvest_scenarios_from_jsonl",
    "main",
    "source_run_split",
]


def _require_supported_mask_version(dto: JsonObject) -> None:
    """Accept only the mask version paired with this hard-cutover wire DTO."""
    raw_version = dto.get("mask_version")
    if raw_version != _MASK_VERSION:
        raise ValueError(
            "scenario_harvest requires masked_emulator_dto mask_version exactly '1.2'; "
            f"got {raw_version!r}. Other versions are a different wire contract and "
            "must not be interpreted optimistically."
        )


def _card_instance(card: JsonObject) -> JsonObject:
    """Build one full-fidelity structured card instance from a public card record."""
    instance: JsonObject = {
        "card_id": card["id"],
        "is_upgraded": bool(card.get("upgraded", False)),
    }
    upgrade_level = card.get("upgradeLevel")
    if upgrade_level is not None:
        instance["upgrade_level"] = int(upgrade_level)
    if card.get("tinkerTimeType"):
        instance["tinker_time_type"] = card["tinkerTimeType"]
    if card.get("tinkerTimeRider"):
        instance["tinker_time_rider"] = card["tinkerTimeRider"]
    enchantment = card.get("enchantment")
    if enchantment:
        scenario_enchantment: JsonObject = {
            "id": enchantment["id"],
            "amount": enchantment.get("amount", 1),
        }
        if enchantment.get("status") is not None:
            scenario_enchantment["status"] = enchantment["status"]
        instance["enchantment"] = scenario_enchantment
    return instance


def _power_stack(power: JsonObject) -> JsonObject | None:
    power_id = power.get("id")
    if power_id in _GOD_MODE_POWER_IDS:
        return None
    stack: JsonObject = {"power_id": power_id, "amount": power["amount"]}
    associated_card = power.get("associatedCard")
    if associated_card:
        stack["associated_card"] = _card_instance(associated_card)
    return stack


def _power_stacks(powers: Any) -> list[JsonObject]:
    result: list[JsonObject] = []
    for power in powers or []:
        stack = _power_stack(power)
        if stack is not None:
            result.append(stack)
    return result


def _expand_multiset_cards(multiset: Any) -> list[JsonObject]:
    """Expand a mask_version 1.2 pile multiset while preserving per-card state."""
    result: list[JsonObject] = []
    for entry in multiset or []:
        if not isinstance(entry, dict):
            continue
        result.extend([_card_instance(entry)] * int(entry.get("count", 0)))
    return result


def _enemy_scenario(enemy: JsonObject) -> JsonObject | None:
    if not enemy.get("isAlive", True):
        return None
    spec: JsonObject = {"monster_id": enemy["id"], "hp": max(1, int(enemy["hp"]))}
    if enemy.get("maxHp") is not None:
        spec["max_hp"] = int(enemy["maxHp"])
    if enemy.get("block"):
        spec["block"] = int(enemy["block"])
    if enemy.get("slotName") is not None:
        spec["slot_name"] = enemy["slotName"]
    intent = enemy.get("intent") or {}
    if intent.get("stateId"):
        spec["forced_move"] = intent["stateId"]
    if enemy.get("stateLog"):
        spec["state_log"] = list(enemy["stateLog"])
    powers = _power_stacks(enemy.get("powers"))
    if powers:
        spec["powers"] = powers
    return spec


def dto_to_scenario_spec(dto: JsonObject, *, seed: int) -> JsonObject | None:
    """Convert one mask_version 1.2 combat-start DTO into a scenario spec."""
    _require_supported_mask_version(dto)
    if dto.get("pendingChoice"):
        return None
    enemies = [
        spec for spec in (_enemy_scenario(e) for e in dto.get("enemies") or []) if spec is not None
    ]
    if not enemies:
        return None

    spec: JsonObject = {
        "character_id": dto["characterId"],
        "player_hp": max(1, int(dto["hp"])),
        "player_max_hp": int(dto["maxHp"]),
        "player_block": int(dto.get("block") or 0),
        "hand": [],
        "draw_pile": [],
        "discard_pile": [],
        "exhaust_pile": [],
        "relics": [r["id"] for r in dto.get("relics") or []],
        "potions": [
            {"slot": index, "potion_id": potion["id"]}
            for index, potion in enumerate(dto.get("potions") or [])
            if potion
        ],
        "player_powers": _power_stacks(dto.get("playerPowers")),
        "seed": seed,
        "enemies": enemies,
    }
    if dto.get("energy") is not None:
        spec["energy"] = int(dto["energy"])
    if dto.get("stars") is not None:
        spec["stars"] = int(dto["stars"])

    extra: JsonObject = {}
    hand_cards = [_card_instance(c) for c in dto.get("hand") or []]
    if hand_cards:
        extra["hand_cards"] = hand_cards
    draw_pile_cards = _expand_multiset_cards(dto.get("drawPile"))
    if draw_pile_cards:
        extra["draw_pile_cards"] = draw_pile_cards
    discard_pile_cards = _expand_multiset_cards(dto.get("discardPile"))
    if discard_pile_cards:
        extra["discard_pile_cards"] = discard_pile_cards
    exhaust_pile_cards = _expand_multiset_cards(dto.get("exhaustPile"))
    if exhaust_pile_cards:
        extra["exhaust_pile_cards"] = exhaust_pile_cards
    orb_slots = dto.get("orbSlots")
    if orb_slots is not None:
        extra["orb_slots"] = int(orb_slots)
    orbs = dto.get("orbs")
    if orbs:
        extra["orbs"] = [{"orb_id": orb["id"]} for orb in orbs if orb.get("id") is not None]
    if dto.get("stepIndex") is not None:
        extra["step_index"] = int(dto["stepIndex"])
    if extra:
        spec["extra"] = extra
    return spec


def _is_combat_start(dto: JsonObject) -> bool:
    return (
        dto.get("currentRoomType") == "CombatRoom"
        and dto.get("boundary") == "stable"
        and dto.get("turnNumber") == 1
        and dto.get("combatRoundNumber") == 1
    )


def _room_key(dto: JsonObject) -> Any:
    room_context = dto.get("room_context") or {}
    return (room_context.get("column"), room_context.get("row"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_split_fractions(*, val_fraction: float, test_fraction: float) -> None:
    for name, value in (("val_fraction", val_fraction), ("test_fraction", test_fraction)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        if not 0.0 <= float(value) < 1.0:
            raise ValueError(f"{name} must be in [0, 1)")
    if float(val_fraction) + float(test_fraction) >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")


def source_run_split(
    source_run_sha256: str,
    *,
    split_seed: int = 0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> str:
    """Assign one source run to train/val/test deterministically.

    Every combat harvested from the same JSONL has the same ``source_run_sha256`` and
    therefore the same split. This prevents sibling combats from leaking across dataset
    roles even if scenario filenames are shuffled independently later.
    """
    if not isinstance(source_run_sha256, str) or len(source_run_sha256) != 64:
        raise ValueError("source_run_sha256 must be a 64-character SHA-256 hex string")
    try:
        bytes.fromhex(source_run_sha256)
    except ValueError as exc:
        raise ValueError("source_run_sha256 must be valid hexadecimal") from exc
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise ValueError("split_seed must be an integer")
    _validate_split_fractions(val_fraction=val_fraction, test_fraction=test_fraction)
    digest = hashlib.sha256(f"{split_seed}:{source_run_sha256}".encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64)
    if unit < float(test_fraction):
        return "test"
    if unit < float(test_fraction) + float(val_fraction):
        return "val"
    return "train"


def is_completed_run_log(path: Path) -> bool:
    """Whether path ends with a valid self_play_run_result record."""
    last_record: JsonObject | None = None
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            last_record = json.loads(line)
    return isinstance(last_record, dict) and last_record.get("event") == "self_play_run_result"


def harvest_scenarios_auto(path: Path, *, rng: random.Random | None = None) -> list[JsonObject]:
    return harvest_scenarios_from_jsonl(
        path, exclude_final_combat=not is_completed_run_log(path), rng=rng
    )


def harvest_scenario_records_from_jsonl(
    path: Path,
    *,
    exclude_final_combat: bool,
    rng: random.Random | None = None,
) -> list[JsonObject]:
    """Harvest scenarios with source-run provenance kept outside the wire payload."""
    rng = rng if rng is not None else random.Random()
    combat_start_dtos: list[JsonObject] = []
    last_room_key: Any = None
    run_result: JsonObject | None = None
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("event") == "self_play_run_result":
                run_result = record
                continue
            if record.get("event") != "selection":
                continue
            dto = (record.get("received") or {}).get("masked_emulator_dto")
            if not isinstance(dto, dict):
                continue
            if dto.get("currentRoomType") != "CombatRoom":
                last_room_key = None
                continue
            room_key = _room_key(dto)
            if room_key != last_room_key and _is_combat_start(dto):
                combat_start_dtos.append(dto)
            last_room_key = room_key

    if exclude_final_combat and combat_start_dtos:
        combat_start_dtos = combat_start_dtos[:-1]

    source_sha = _file_sha256(path)
    completed = is_completed_run_log(path)
    god_mode_raw = None if run_result is None else run_result.get("god_mode")
    god_mode = god_mode_raw if isinstance(god_mode_raw, bool) else None
    if god_mode is True:
        collection_mode = "god_mode_coverage"
        dataset_role = "coverage_pretraining"
        promotion_eligible = False
    elif god_mode is False:
        collection_mode = "normal_policy"
        dataset_role = "normal_policy"
        promotion_eligible = completed
    else:
        collection_mode = "unknown"
        dataset_role = "unknown"
        promotion_eligible = False

    records: list[JsonObject] = []
    for combat_index, dto in enumerate(combat_start_dtos):
        spec = dto_to_scenario_spec(dto, seed=rng.randint(1, 2**31 - 1))
        if spec is None:
            continue
        room_context = dto.get("room_context")
        provenance: JsonObject = {
            "source_run_sha256": source_sha,
            "source_run_id": None if run_result is None else run_result.get("run_id"),
            "original_run_seed": None if run_result is None else run_result.get("seed"),
            "source_completed": completed,
            "god_mode": god_mode,
            "collection_mode": collection_mode,
            "dataset_role": dataset_role,
            "promotion_eligible": promotion_eligible,
            "combat_index": combat_index,
            "step_index": dto.get("stepIndex"),
            "room_context": dict(room_context) if isinstance(room_context, dict) else {},
        }
        records.append({"scenario": spec, "provenance": provenance})
    return records


def harvest_scenarios_from_jsonl(
    path: Path, *, exclude_final_combat: bool, rng: random.Random | None = None
) -> list[JsonObject]:
    return [
        dict(record["scenario"])
        for record in harvest_scenario_records_from_jsonl(
            path,
            exclude_final_combat=exclude_final_combat,
            rng=rng,
        )
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for per-scenario seed assignment")
    parser.add_argument("--split-seed", type=int, default=0, help="deterministic source-run split seed")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_split_fractions(val_fraction=args.val_fraction, test_fraction=args.test_fraction)
    rng = random.Random(args.seed)
    input_paths = sorted(args.input_dir.glob("*.jsonl"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_scenarios = 0
    incomplete_files = 0
    manifest_entries: list[JsonObject] = []
    for input_path in input_paths:
        completed = is_completed_run_log(input_path)
        if not completed:
            incomplete_files += 1
        records = harvest_scenario_records_from_jsonl(
            input_path, exclude_final_combat=not completed, rng=rng
        )
        for index, record in enumerate(records):
            spec = record["scenario"]
            provenance = dict(record["provenance"])
            output_path = args.output_dir / f"{input_path.stem}-combat{index:02d}.json"
            output_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
            provenance["scenario"] = output_path.name
            provenance["split"] = source_run_split(
                provenance["source_run_sha256"],
                split_seed=args.split_seed,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
            )
            manifest_entries.append(provenance)
        total_scenarios += len(records)

    manifest = {
        "manifest_schema_version": HARVEST_MANIFEST_SCHEMA_VERSION,
        "split_unit": "source_run_sha256",
        "split_seed": args.split_seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "scenarios": manifest_entries,
    }
    (args.output_dir / "harvest_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "files_processed": len(input_paths),
        "incomplete_files": incomplete_files,
        "scenarios_written": total_scenarios,
        "manifest": str(args.output_dir / "harvest_manifest.json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())