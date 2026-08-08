"""`instance_config` builders for the three top-level ways to start an instance.

Each dataclass here maps to exactly one `start_instance` entry point in this package
(see `how_to_use.md`):

- `CombatScenario`  -> `start_combat_from_state.start_combat_from_state()`
- `RunSnapshot`      -> `start_run_from_state.start_run_from_state()`
- `NewRunConfig`     -> `start_new_run.start_new_run()`

`CombatScenario`/`RunSnapshot` intentionally leave their board-defining fields with NO
default value, so constructing one without a complete board is a `TypeError` at the call
site rather than a silently-incomplete scenario reaching RL. `NewRunConfig` is the
opposite case (a normal game start has no board to supply) and defaults accordingly.

Field coverage for `CombatScenario`/`EnemyScenario` mirrors STS2_RL's
`Combat/battle_emulator.py:build_scenario_from_spec()` (see
`Common/schemas/combat_scenario_input_schema.json` there for the authoritative list).
Only the fields load-bearing enough to matter for deck/board evaluation are modeled
directly; anything else that schema accepts (pending_choice, per-card upgrade info,
orb slot sizing, ...) can still be supplied via `extra`, which is merged in verbatim -
this dataclass is deliberately not a hard gate on RL's schema evolving further.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class EnemyScenario:
    """One living enemy in a `CombatScenario`. `monster_id`/`hp` are the two fields
    `build_scenario_from_spec` itself hard-requires per enemy; everything else there
    is optional and defaults sensibly on the RL side when omitted."""

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
    """A complete Combat board state. Required fields are exactly the ones a
    deck/board-evaluation-aware decision needs to not be silently working from a
    partial picture: who's playing, their HP, their full deck (as three piles), and
    the enemies they're facing. Optional fields are pieces of state that can
    legitimately be empty/absent (no potions carried, no active powers, ...) rather
    than missing information.
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
    # per-card upgrade info via hand_cards/draw_pile_cards/..., shuffle_rng_seed,
    # step_index, ...) - merged into the built instance_config verbatim, letting
    # the caller supply anything build_scenario_from_spec accepts without this
    # dataclass having to track its schema 1:1.
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.enemies:
            raise ValueError("CombatScenario.enemies must not be empty (no living enemies)")

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
    """A complete Whole Run board state to resume from.

    `snapshot_json` is the serialized form `WholeRunSession.save_state()` produces on
    the RL side (a full run snapshot: map position, deck, relics, gold, HP, act/floor
    progress, ...) - this dataclass does not re-model any of that shape itself,
    it just carries the opaque blob plus the identity fields RL needs regardless.

    KNOWN GAP (as of this writing): `API/instance_whole_run.py`'s `WholeRunInstance`
    does not yet consume a snapshot field from `instance_config` - it always calls
    `WholeRunSession.start_run(seed, character_id, ascension)`, i.e. a FRESH run,
    even though `WholeRunSession.load_state(snapshot_json)` already exists and could
    be wired to it. `start_run_from_state.start_run_from_state()` refuses to run
    until that RL-side wiring lands, specifically so this never silently starts a
    fresh run while claiming to resume a specific one - see that module's docstring.
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
    to supply. `seed=None` here means "omit it and let the caller/RL decide" - this
    dataclass is a pure mapping and does not itself invent a seed. It is
    `start_new_run.start_new_run()` that fills in a fresh random seed by default
    when the caller doesn't pin one, so each normal-start call produces a different
    run (see that module for why the randomness policy lives there, not here).
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
