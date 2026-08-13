"""`instance_config` builders for the two top-level ways to start an instance:
`CombatScenario` and `NewRunConfig` (see `how_to_use.md`).

`CombatScenario` leaves its board-defining fields with no default, so an incomplete
scenario is a `TypeError` at the call site rather than a silently-incomplete request
to RL. `NewRunConfig` has no board to supply and defaults accordingly.

`CombatScenario`/`EnemyScenario` serialize to STS2_RL's
`Combat/battle_emulator.py:build_scenario_from_spec()` input shape. Convenience forms
for potion slots and power stacks are normalized to the wire shape before the config is
sent. Anything not modeled directly (pending_choice, per-card upgrades, ...) can still
go through `extra`, merged in verbatim. `extra` may not override fields modeled
explicitly by `CombatScenario` or the fixed `instance_type` discriminator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

JsonObject = dict[str, Any]
PotionSpec = str | Mapping[str, Any] | None
PowerStacks = Mapping[str, Any] | Sequence[Mapping[str, Any]]


def _serialize_potions(potions: Sequence[PotionSpec]) -> list[JsonObject]:
    """Normalize potion-belt shorthand to RL's ``{slot, potion_id}`` records.

    Strings use their sequence index as the slot and ``None`` represents an empty slot.
    Mapping entries are copied verbatim for callers that already have RL-shaped potion
    records.
    """
    result: list[JsonObject] = []
    for slot, potion in enumerate(potions):
        if potion is None:
            continue
        if isinstance(potion, str):
            result.append({"slot": slot, "potion_id": potion})
            continue
        if isinstance(potion, Mapping):
            result.append(dict(potion))
            continue
        raise TypeError("CombatScenario.potions entries must be strings, mappings, or None")
    return result


def _serialize_power_stacks(powers: PowerStacks) -> list[JsonObject]:
    """Normalize ``{power_id: amount}`` shorthand to RL power-stack records.

    A sequence of mappings is already the exact/advanced form and is copied verbatim.
    A single ``{power_id/id, amount}`` mapping is also accepted as one exact stack.
    """
    if isinstance(powers, Mapping):
        if "amount" in powers and ("power_id" in powers or "id" in powers):
            return [dict(powers)]
        return [
            {"power_id": str(power_id), "amount": amount}
            for power_id, amount in powers.items()
        ]
    return [dict(power) for power in powers]


@dataclass(frozen=True)
class EnemyScenario:
    """One living enemy in a `CombatScenario`.

    ``powers={"STRENGTH": 2}`` is a convenience shorthand; a sequence of RL-shaped
    power-stack mappings is accepted when associated-card or other exact stack state is
    needed. ``forced_move`` and ``state_log`` preserve the enemy move-machine state that
    RL's ``build_scenario_from_spec()`` can restore exactly.
    """

    monster_id: str
    hp: int
    max_hp: int | None = None
    block: int = 0
    slot_name: str | None = None
    forced_move: str | None = None
    state_log: Sequence[str] = field(default_factory=tuple)
    frog_knight_has_beetle_charged: bool | None = None
    waterfall_giant_current_pressure_gun_damage: int | None = None
    powers: PowerStacks = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        payload: JsonObject = {"monster_id": self.monster_id, "hp": self.hp}
        if self.max_hp is not None:
            payload["max_hp"] = self.max_hp
        if self.block:
            payload["block"] = self.block
        if self.slot_name is not None:
            payload["slot_name"] = self.slot_name
        if self.forced_move is not None:
            payload["forced_move"] = self.forced_move
        if self.state_log:
            payload["state_log"] = list(self.state_log)
        if self.frog_knight_has_beetle_charged is not None:
            payload["frog_knight_has_beetle_charged"] = self.frog_knight_has_beetle_charged
        if self.waterfall_giant_current_pressure_gun_damage is not None:
            payload["waterfall_giant_current_pressure_gun_damage"] = (
                self.waterfall_giant_current_pressure_gun_damage
            )
        if self.powers:
            payload["powers"] = _serialize_power_stacks(self.powers)
        return payload


@dataclass(frozen=True)
class CombatScenario:
    """A complete Combat board state. Required fields are what a deck/board-aware
    decision needs to not work from a partial picture: who's playing, their HP, full
    deck (as three piles), and the enemies. Optional fields are state that can
    legitimately be empty rather than missing.

    ``potions`` accepts belt-order string IDs (use ``None`` for an empty slot) or
    already-shaped ``{slot, potion_id}`` mappings. ``player_powers`` accepts a simple
    ``{power_id: amount}`` shorthand or a sequence of RL-shaped power-stack mappings.
    """

    character_id: str
    player_hp: int
    player_max_hp: int
    hand: Sequence[str]
    draw_pile: Sequence[str]
    discard_pile: Sequence[str]
    enemies: Sequence[EnemyScenario]

    exhaust_pile: Sequence[str] = field(default_factory=tuple)
    player_block: int = 0
    energy: int | None = None
    stars: int | None = None
    relics: Sequence[str] = field(default_factory=tuple)
    potions: Sequence[PotionSpec] = field(default_factory=tuple)
    player_powers: PowerStacks = field(default_factory=dict)
    seed: int = 1
    # Escape hatch for schema fields not modeled above (orbs, pending_choice,
    # per-card upgrades, ...) - merged into instance_config verbatim.
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.enemies:
            raise ValueError("CombatScenario.enemies must not be empty (no living enemies)")

        modeled_keys = {item.name for item in fields(self) if item.name != "extra"}
        overlap = modeled_keys.intersection(self.extra)
        if "instance_type" in self.extra:
            overlap.add("instance_type")
        if overlap:
            raise ValueError(
                "CombatScenario.extra is only for unmodeled fields and must not override: "
                f"{sorted(overlap)}"
            )

    def to_instance_config(self) -> JsonObject:
        return {
            "instance_type": "combat",
            "character_id": self.character_id,
            "player_hp": self.player_hp,
            "player_max_hp": self.player_max_hp,
            "player_block": self.player_block,
            "hand": list(self.hand),
            "draw_pile": list(self.draw_pile),
            "discard_pile": list(self.discard_pile),
            "exhaust_pile": list(self.exhaust_pile),
            "relics": list(self.relics),
            "potions": _serialize_potions(self.potions),
            "player_powers": _serialize_power_stacks(self.player_powers),
            "enemies": [e.to_dict() for e in self.enemies],
            "seed": self.seed,
            **({"energy": self.energy} if self.energy is not None else {}),
            **({"stars": self.stars} if self.stars is not None else {}),
            **dict(self.extra),
        }


@dataclass(frozen=True)
class NewRunConfig:
    """A normal, from-scratch game start with no board to supply. `seed=None` means
    omit it; `start_new_run()` is what fills in a fresh random seed by default, not
    this dataclass (kept a pure mapping).
    """

    character_id: str
    ascension: int = 0
    seed: int | None = None
    god_mode: bool = False
    """Explicit, per-instance opt-in for RL's God Mode data-collection support (see
    STS2_RL#39/#40) - the player is invincible for the whole run. Default False; never
    silently enabled. Callers that set this are responsible for keeping the resulting
    data separate from normal collection (see `runner.self_play`'s `--god-mode`)."""

    def to_instance_config(self) -> JsonObject:
        return {
            "instance_type": "whole_run",
            "character_id": self.character_id,
            "ascension": self.ascension,
            **({"seed": self.seed} if self.seed is not None else {}),
            **({"god_mode": True} if self.god_mode else {}),
        }
