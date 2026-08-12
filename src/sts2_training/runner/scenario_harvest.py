"""Harvest `CombatScenario` JSON specs (Combat starting states) from collected
self-play JSONL logs, for the combat-search-learning pipeline's `--scenario` input
(see `oracle_collection.py`, `docs/combat_search_learning_plan.md`).

Mirrors STS2_RL's `Combat/battle_emulator.py::build_scenario_from_state()` field
mapping (the canonical `masked_emulator_dto` -> `CombatScenario` conversion used
in-process for restore/lookahead), but in pure Python producing the on-disk JSON
spec shape `oracle_collection.py`'s `_scenario_from_json()` already loads
(`CombatScenario(**data)`, `EnemyScenario(**enemy)`). Every pile (`hand` and the
`drawPile`/`discardPile`/`exhaustPile` multisets, mask_version >= 1.2) goes through
the `extra` escape hatch as `hand_cards`/`draw_pile_cards`/`discard_pile_cards`/
`exhaust_pile_cards`/`orbs`/`orb_slots`, matching `build_scenario_from_spec()`'s
accepted structured input - card identity (upgrade level, enchantment, Mad Science
tinker-time state) is preserved for all four piles; only pile ORDER is genuinely lost
(Hidden Information, masked server-side before this module ever sees it).

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
_MIN_MASK_VERSION = (1, 2)

__all__ = ["dto_to_scenario_spec", "harvest_scenarios_from_jsonl", "main"]


def _parse_mask_version(value: Any) -> tuple[int, ...] | None:
    """Parse dotted numeric mask versions without treating e.g. 1.10 as float 1.1."""
    if not isinstance(value, (str, int, float)):
        return None
    parts = str(value).split(".")
    if not parts or any(not part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _require_supported_mask_version(dto: JsonObject) -> None:
    """Reject DTOs whose pile shape predates the full-card-identity multiset contract."""
    raw_version = dto.get("mask_version")
    version = _parse_mask_version(raw_version)
    if version is None or version < _MIN_MASK_VERSION:
        raise ValueError(
            "scenario_harvest requires masked_emulator_dto mask_version >= 1.2; "
            f"got {raw_version!r}. Older/missing versions use a different pile multiset "
            "shape and cannot be restored without losing card state."
        )


def _card_instance(card: JsonObject) -> JsonObject:
    """Build a `{"card_id","is_upgraded",...}` structured card instance from any DTO
    card dict - both a `hand` entry and a `drawPile`/`discardPile`/`exhaustPile`
    multiset record (API/masking.py's `_multiset_of`, mask_version >= 1.2) share this
    shape (id/upgraded/upgradeLevel/tinkerTimeType/tinkerTimeRider/enchantment), so one
    function builds a scenario-ready instance from either."""
    instance: JsonObject = {
        "card_id": card["id"],
        "is_upgraded": bool(card.get("upgraded", False)),
    }
    upgrade_level = card.get("upgradeLevel")
    if upgrade_level:
        instance["upgrade_level"] = int(upgrade_level)
    if card.get("tinkerTimeType"):
        instance["tinker_time_type"] = card["tinkerTimeType"]
    if card.get("tinkerTimeRider"):
        instance["tinker_time_rider"] = card["tinkerTimeRider"]
    enchantment = card.get("enchantment")
    if enchantment:
        instance["enchantment"] = {
            "id": enchantment["id"],
            "amount": enchantment.get("amount", 1),
        }
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
    """Expand a masked-pile multiset (API/masking.py's `_multiset_of`, mask_version
    >= 1.2) into full-fidelity `_card_instance` records, repeated `count` times each.
    Pile order is genuinely Hidden Information and stays lost - per-card upgrade level
    and enchantment state are recovered (mask_version < 1.2 exposed neither)."""
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
    powers = _power_stacks(enemy.get("powers"))
    if powers:
        spec["powers"] = powers
    return spec


def dto_to_scenario_spec(dto: JsonObject, *, seed: int) -> JsonObject | None:
    """Convert one combat-start `masked_emulator_dto` into an on-disk scenario spec.

    Requires `mask_version >= 1.2`, the first masked-pile representation that carries
    full per-card identity instead of the legacy `{card_id: count}` shape. Returns
    `None` when the snapshot isn't safe/complete to harvest (a live `pendingChoice`,
    or no living enemies - a state that should already be terminal/transitioning rather
    than a real starting point).
    """
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

    # "hand" is exposed as a full ordered list of card dicts; "drawPile"/"discardPile"/
    # "exhaustPile" are masked down to per-distinct-instance multisets (API/masking.py's
    # _multiset_of, mask_version >= 1.2) - pile ORDER is genuinely lost either way
    # (Hidden Information), but upgrade/enchantment state is preserved for both via the
    # upgrade-preserving *_cards extra fields; plain draw_pile/discard_pile/exhaust_pile
    # stay empty (never both populated for the same pile - see CombatScenario.HandCards
    # / GameInstance.ResolveCardSpecs's doc comments).
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
