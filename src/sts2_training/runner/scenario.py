"""`instance_config` builders for the three top-level ways to start an instance:
`CombatScenario`, `RunSnapshot`, `NewRunConfig` (see `how_to_use.md`).

`CombatScenario`/`RunSnapshot` leave their board-defining fields with no default, so
an incomplete scenario is a `TypeError` at the call site rather than a
silently-incomplete request to RL. `NewRunConfig` has no board to supply and defaults
accordingly.

`CombatScenario`/`EnemyScenario` field coverage mirrors STS2_RL's
`Combat/battle_emulator.py:build_scenario_from_spec()`; anything not modeled directly
(pending_choice, per-card upgrades, ...) can still go through `extra`, merged in
verbatim. `extra` may not override fields modeled explicitly by `CombatScenario` or
the fixed `instance_type` discriminator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class EnemyScenario:
    """One living enemy in a `CombatScenario`. `monster_id`/`hp` are the only fields
    `build_scenario_from_spec` requires per enemy; the rest default on the RL side."""

    monster_id: str
    hp: int
    max_hp: int | None = None
    block: int = 0
    powers: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        payload: JsonObject = {"monster_id": self.monster_id, "hp": self.hp}
        if self.max_hp is not None:
            payload["max_hp"] = self.max_hp
        if self.block:
            payload["block"] = self.block
        if self.powers:
            payload["powers"] = dict(self.powers)
        return payload


@dataclass(frozen=True)
class CombatScenario:
    """A complete Combat board state. Required fields are what a deck/board-aware
    decision needs to not work from a partial picture: who's playing, their HP, full
    deck (as three piles), and the enemies. Optional fields are state that can
    legitimately be empty rather than missing.
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
    potions: Sequence[str] = field(default_factory=tuple)
    player_powers: Mapping[str, int] = field(default_factory=dict)
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
            "potions": list(self.potions),
            "player_powers": dict(self.player_powers),
            "enemies": [e.to_dict() for e in self.enemies],
            "seed": self.seed,
            **({"energy": self.energy} if self.energy is not None else {}),
            **({"stars": self.stars} if self.stars is not None else {}),
            **dict(self.extra),
        }


@dataclass(frozen=True)
class RunSnapshot:
    """A complete Whole Run board state to resume from. `snapshot_json` is
    `WholeRunSession.save_state()`'s opaque output on the RL side - this dataclass
    just carries it plus the identity fields RL needs, not its shape.

    KNOWN GAP: `WholeRunInstance` doesn't consume a snapshot field yet - it always
    starts a fresh run. `start_run_from_state()` refuses to run until that lands
    rather than silently starting fresh under a "resumed" label - see that module.
    """

    character_id: str
    ascension: int
    seed: int
    snapshot_json: str

    def __post_init__(self) -> None:
        if not self.snapshot_json:
            raise ValueError("RunSnapshot.snapshot_json must not be empty")

    def to_instance_config(self) -> JsonObject:
        return {
            "instance_type": "whole_run",
            "character_id": self.character_id,
            "ascension": self.ascension,
            "seed": self.seed,
            "snapshot_json": self.snapshot_json,
        }


@dataclass(frozen=True)
class NewRunConfig:
    """A normal, from-scratch game start - the only one of the three with no board
    to supply. `seed=None` means omit it; `start_new_run()` is what fills in a fresh
    random seed by default, not this dataclass (kept a pure mapping).
    """

    character_id: str
    ascension: int = 0
    seed: int | None = None

    def to_instance_config(self) -> JsonObject:
        return {
            "instance_type": "whole_run",
            "character_id": self.character_id,
            "ascension": self.ascension,
            **({"seed": self.seed} if self.seed is not None else {}),
        }
