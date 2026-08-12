"""Harvest `CombatScenario` JSON specs (Combat starting states) from collected
self-play JSONL logs, for the combat-search-learning pipeline's `--scenario` input
(see `oracle_collection.py`, `docs/combat_search_learning_plan.md`).

Mirrors STS2_RL's `Combat/battle_emulator.py::build_scenario_from_state()` field
mapping (the canonical `masked_emulator_dto` -> `CombatScenario` conversion used
in-process for restore/lookahead), but in pure Python producing the on-disk JSON
spec shape `oracle_collection.py`'s `_scenario_from_json()` already loads
(`CombatScenario(**data)`, `EnemyScenario(**enemy)` - card-pile order/upgrade state
and orbs go through the `extra` escape hatch as `hand_cards`/`draw_pile_cards`/
`discard_pile_cards`/`exhaust_pile_cards`/`orbs`/`orb_slots`, matching
`build_scenario_from_spec()`'s accepted structured input).

`player_powers`/enemy `powers` always drop the three GOD MODE powers
(`STRENGTH_POWER`/`BUFFER_POWER`/`REGEN_POWER`) regardless of whether the source
file was already processed by `god_mode_correction` - this module is the sole
place responsible for that when harvesting directly from raw (uncorrected) logs,
so it does not rely on file-level correction having already run.

A "combat start" is the first `stable`-boundary decision (turn 1, round 1) after
entering a room whose `currentRoomType` is `CombatRoom` - detected by watching the
`(column, row)` room-context change, not by boundary transition alone, since a
room can be re-entered in principle. A snapshot with a live `pendingChoice` or no
living enemies is skipped as unsafe/incomplete to harvest.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]

# Mirrors god_mode_correction.GOD_MODE_POWER_IDS - kept separate rather than
# imported so this module has no dependency on that tool's presence/API shape.
_GOD_MODE_POWER_IDS = frozenset({"STRENGTH_POWER", "BUFFER_POWER", "REGEN_POWER"})

__all__ = ["dto_to_scenario_spec", "harvest_scenarios_from_jsonl", "main"]


def _card_instance(card: JsonObject) -> JsonObject:
    instance: JsonObject = {
        "card_id": card["id"],
        "is_upgraded": bool(card.get("upgraded", False)),
    }
    if card.get("tinkerTimeType"):
        instance["tinker_time_type"] = card["tinkerTimeType"]
    if card.get("tinkerTimeRider"):
        instance["tinker_time_rider"] = card["tinkerTimeRider"]
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


def _expand_multiset(multiset: Any) -> list[str]:
    """Expand a `{card_id: count}` masked-pile multiset back into a plain id list."""
    result: list[str] = []
    for card_id, count in (multiset or {}).items():
        result.extend([card_id] * int(count))
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
    powers = _power_stacks(enemy.get("powers"))
    if powers:
        spec["powers"] = powers
    return spec


def dto_to_scenario_spec(dto: JsonObject, *, seed: int) -> JsonObject | None:
    """Convert one combat-start `masked_emulator_dto` into an on-disk scenario spec.

    Returns `None` when the snapshot isn't safe/complete to harvest (a live
    `pendingChoice`, or no living enemies - a state that should already be
    terminal/transitioning rather than a real starting point).
    """
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

    # "hand" is exposed as a full ordered list of card dicts (upgrade info intact);
    # "drawPile"/"discardPile"/"exhaustPile" are masked down to {card_id: count}
    # multisets (API/masking.py's _MULTISET_PILE_KEYS) - order and per-card upgrade
    # state for cards sitting in those three piles is not recoverable from the DTO,
    # so they go through the plain id-list fields (repeated `count` times) instead
    # of the upgrade-preserving *_cards extra used for hand.
    spec["draw_pile"] = _expand_multiset(dto.get("drawPile"))
    spec["discard_pile"] = _expand_multiset(dto.get("discardPile"))
    spec["exhaust_pile"] = _expand_multiset(dto.get("exhaustPile"))

    extra: JsonObject = {}
    hand_cards = [_card_instance(c) for c in dto.get("hand") or []]
    if hand_cards:
        extra["hand_cards"] = hand_cards
    orb_slots = dto.get("orbSlots")
    if orb_slots is not None:
        extra["orb_slots"] = int(orb_slots)
    orbs = dto.get("orbs")
    if orbs:
        extra["orbs"] = [
            {"orb_id": orb["id"]} for orb in orbs if orb.get("id") is not None
        ]
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


def is_completed_run_log(path: Path) -> bool:
    """Whether `path` ends with a valid `self_play_run_result` record.

    Mirrors `god_mode_correction._is_god_mode_run_result`'s validity check (a
    `self_play_run_result` with `god_mode: true`), generalized to also accept a
    non-god-mode successful run (`god_mode` absent/false is fine as long as the
    terminal record is present) - completeness, not god-mode-ness, is what decides
    whether every detected combat in the file is safe to harvest.
    """
    last_record: JsonObject | None = None
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            last_record = json.loads(line)
    return isinstance(last_record, dict) and last_record.get("event") == "self_play_run_result"


def harvest_scenarios_auto(path: Path, *, rng: random.Random | None = None) -> list[JsonObject]:
    """Harvest scenarios from `path`, excluding the final combat iff the run never
    completed - see `is_completed_run_log`."""
    return harvest_scenarios_from_jsonl(
        path, exclude_final_combat=not is_completed_run_log(path), rng=rng
    )


def harvest_scenarios_from_jsonl(
    path: Path, *, exclude_final_combat: bool, rng: random.Random | None = None
) -> list[JsonObject]:
    """Harvest every combat-start scenario spec found in one collected JSONL log.

    `exclude_final_combat=True` drops the last detected combat start - for a run
    whose log ends mid-failure, that final combat is the one that failed and must
    not be treated as a valid, resolvable starting state.
    """
    rng = rng if rng is not None else random.Random()
    combat_start_dtos: list[JsonObject] = []
    last_room_key: Any = None
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
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

    specs: list[JsonObject] = []
    for dto in combat_start_dtos:
        spec = dto_to_scenario_spec(dto, seed=rng.randint(1, 2**31 - 1))
        if spec is not None:
            specs.append(spec)
    return specs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for per-scenario seed assignment")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rng = random.Random(args.seed)
    input_paths = sorted(args.input_dir.glob("*.jsonl"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_scenarios = 0
    incomplete_files = 0
    for input_path in input_paths:
        completed = is_completed_run_log(input_path)
        if not completed:
            incomplete_files += 1
        specs = harvest_scenarios_from_jsonl(
            input_path, exclude_final_combat=not completed, rng=rng
        )
        for index, spec in enumerate(specs):
            output_path = args.output_dir / f"{input_path.stem}-combat{index:02d}.json"
            output_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        total_scenarios += len(specs)

    print(
        json.dumps(
            {
                "files_processed": len(input_paths),
                "incomplete_files": incomplete_files,
                "scenarios_written": total_scenarios,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
