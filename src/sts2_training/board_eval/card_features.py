"""Per-card feature extraction for the Run-state board evaluation model."""

from __future__ import annotations

import csv
import enum
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DEFAULT_CARD_FEATURES_CSV = (
    Path(__file__).resolve().parents[3] / "tools" / "output" / "card_secondary_features.csv"
)

_AOE_TARGET_TYPES = frozenset({"AllEnemies", "AllAllies"})
_TRUE = "TRUE"


class ScalingSource(enum.Enum):
    DECK_COUNT = "DECK_COUNT"
    COMBAT_COUNT = "COMBAT_COUNT"
    TURN_COUNT = "TURN_COUNT"
    PLAYER_VALUE = "PLAYER_VALUE"
    UNPARSED = "UNPARSED"


class Scope(enum.Enum):
    RUN_STATIC = "RUN_STATIC"
    FLOOR = "FLOOR"
    COMBAT = "COMBAT"
    TURN = "TURN"


@dataclass(frozen=True)
class ReferenceScaling:
    """One structural scaling reference retained from the static card data."""

    source: ScalingSource
    scope: Scope
    filter_value: str
    referent: Literal["self", "enemy"] | None = None
    coefficient: float | None = None


@dataclass(frozen=True)
class CardFeatures:
    card_id: str
    upgrade: bool = False

    cost: float = 0.0
    is_x_cost: bool = False
    star_cost: float = 0.0
    damage: float = 0.0
    block: float = 0.0
    draw: float = 0.0
    energy_gain: float = 0.0

    is_attack: bool = False
    is_skill: bool = False
    is_power: bool = False
    is_aoe: bool = False
    is_multi_hit: bool = False
    hit_count_reference: ReferenceScaling | None = None
    exhaust: bool = False
    retain: bool = False
    innate: bool = False
    strength_scaling: bool = False
    dexterity_scaling: bool = False
    poison: bool = False
    discard: bool = False
    exhaust_generation: bool = False

    heal_total: float = 0.0
    max_hp_gain_total: float = 0.0
    gold_gain_total: float = 0.0
    card_generation_count: float = 0.0
    card_generation_uncertain_count: float = 0.0
    card_transform_count: float = 0.0
    card_transform_uncertain_count: float = 0.0
    card_tutor_count: float = 0.0
    card_tutor_uncertain_count: float = 0.0
    buff_apply_total: float = 0.0
    debuff_apply_total: float = 0.0
    orb_generation_count: float = 0.0
    orb_generation_uncertain_count: float = 0.0
    orb_evoke_count: float = 0.0
    character_resource_gain: float = 0.0
    other_known_effect_count: float = 0.0
    unparsed_effect_count: float = 0.0

    dynamic_scalings: tuple[ReferenceScaling, ...] = ()

    card_type: str = ""
    rarity: str = ""
    keywords: tuple[str, ...] = ()
    deck_eval_excluded: bool = False
    exclusion_reason: str = ""

    @property
    def uses_star_cost(self) -> bool:
        return self.star_cost > 0

    @property
    def is_dynamic(self) -> bool:
        return bool(self.dynamic_scalings or self.hit_count_reference)

    def to_vector(self) -> tuple[float, ...]:
        return tuple(_as_number(getattr(self, name)) for name in FEATURE_NAMES)


FEATURE_NAMES: tuple[str, ...] = (
    "upgrade",
    "cost",
    "star_cost",
    "uses_star_cost",
    "damage",
    "block",
    "draw",
    "energy_gain",
    "is_attack",
    "is_skill",
    "is_power",
    "is_aoe",
    "is_multi_hit",
    "exhaust",
    "retain",
    "innate",
    "strength_scaling",
    "dexterity_scaling",
    "poison",
    "discard",
    "exhaust_generation",
    "heal_total",
    "max_hp_gain_total",
    "gold_gain_total",
    "card_generation_count",
    "card_generation_uncertain_count",
    "card_transform_count",
    "card_transform_uncertain_count",
    "card_tutor_count",
    "card_tutor_uncertain_count",
    "buff_apply_total",
    "debuff_apply_total",
    "orb_generation_count",
    "orb_generation_uncertain_count",
    "orb_evoke_count",
    "character_resource_gain",
    "other_known_effect_count",
    "unparsed_effect_count",
    "is_dynamic",
)


def _as_number(value: bool | float | str) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


class UnknownCardError(KeyError):
    def __init__(self, card_id: str) -> None:
        super().__init__(card_id)
        self.card_id = card_id

    def __str__(self) -> str:
        return f"unknown card_id: {self.card_id!r}"


class CardFeatureExtractor:
    def __init__(self, rows: Iterable[Mapping[str, str]]) -> None:
        self._rows_by_id = {row["card_id"]: row for row in rows}

    @classmethod
    def from_csv(cls, path: Path | str = DEFAULT_CARD_FEATURES_CSV) -> CardFeatureExtractor:
        path = Path(path)
        with path.open(encoding="utf-8", newline="") as handle:
            return cls(list(csv.DictReader(handle)))

    def card_ids(self) -> list[str]:
        return list(self._rows_by_id)

    def __contains__(self, card_id: str) -> bool:
        return card_id in self._rows_by_id

    def __len__(self) -> int:
        return len(self._rows_by_id)

    def extract(self, card_id: str, *, upgraded: bool = False) -> CardFeatures:
        try:
            row = self._rows_by_id[card_id]
        except KeyError:
            raise UnknownCardError(card_id) from None
        return _row_to_features(row, upgraded=upgraded)

    def extract_all(self, *, upgraded_ids: Iterable[str] = ()) -> dict[str, CardFeatures]:
        upgraded = set(upgraded_ids)
        return {
            card_id: _row_to_features(row, upgraded=card_id in upgraded)
            for card_id, row in self._rows_by_id.items()
        }


def _row_to_features(row: Mapping[str, str], *, upgraded: bool) -> CardFeatures:
    card_type = row.get("card_type") or ""
    target_type = row.get("target_type") or ""
    keywords = tuple(_split_list(row.get("keywords")))
    keyword_set = set(keywords)

    is_x_cost = _parse_bool(row.get("is_x_cost"))
    cost = -1.0 if is_x_cost else _parse_float(row.get("energy_cost"))
    star_cost = _parse_float(row.get("star_cost"))
    damage = _parse_float(row.get("damage")) + _parse_float(row.get("extra_damage"))
    block = _parse_float(row.get("block"))
    draw = _parse_float(row.get("cards_drawn"))
    energy_gain = _parse_float(row.get("energy"))
    repeat_hits = _parse_float(row.get("repeat_hits"))
    hit_count_reference = _hit_count_reference(row)

    is_attack = card_type == "Attack"
    is_skill = card_type == "Skill"
    is_power = card_type == "Power"

    buffs = _power_entries(row.get("buffs_applied"))
    debuffs = _power_entries(row.get("debuffs_applied"))
    other_powers = _power_entries(row.get("other_powers_applied"))
    poison = any("POISON" in label.upper() for label, _ in (*buffs, *debuffs, *other_powers))

    effects = _effect_family_values(row, buffs=buffs, debuffs=debuffs, other_powers=other_powers)

    return CardFeatures(
        card_id=row["card_id"],
        upgrade=upgraded,
        cost=cost,
        is_x_cost=is_x_cost,
        star_cost=star_cost,
        damage=damage,
        block=block,
        draw=draw,
        energy_gain=energy_gain,
        is_attack=is_attack,
        is_skill=is_skill,
        is_power=is_power,
        is_aoe=target_type in _AOE_TARGET_TYPES,
        is_multi_hit=repeat_hits > 0 or hit_count_reference is not None,
        hit_count_reference=hit_count_reference,
        exhaust="Exhaust" in keyword_set,
        retain="Retain" in keyword_set,
        innate="Innate" in keyword_set,
        strength_scaling=is_attack and damage > 0,
        dexterity_scaling=block > 0,
        poison=poison,
        discard=(
            _parse_bool(row.get("discard_pile_reference"))
            or _parse_bool(row.get("discards_other_cards"))
            or _parse_bool(row.get("sly_keyword"))
        ),
        exhaust_generation=_parse_bool(row.get("exhausts_other_cards")),
        dynamic_scalings=_dynamic_scalings(row),
        card_type=card_type,
        rarity=row.get("rarity") or "",
        keywords=keywords,
        deck_eval_excluded=_parse_bool(row.get("deck_eval_excluded")),
        exclusion_reason=row.get("exclusion_reason") or "",
        **effects,
    )


def _hit_count_reference(row: Mapping[str, str]) -> ReferenceScaling | None:
    """Retain what determines a dynamic multi-hit count without inventing a count.

    Literal ``RepeatVar`` values remain represented only by ``is_multi_hit``.
    For calculated hit counts, keep a structural reference when the source
    snippet is recognizable and preserve the source snippet as UNPARSED
    otherwise. This deliberately does not turn the reference into guessed
    ``effective_damage``.
    """
    snippet = row.get("review_snippet") or ""
    if "CalculatedHits" not in snippet:
        return None

    if "OrbQueue.Orbs.Count" in snippet:
        return ReferenceScaling(
            ScalingSource.COMBAT_COUNT,
            Scope.COMBAT,
            "ORB_COUNT",
        )

    if "PileType.Hand" in snippet and "CardType.Skill" in snippet:
        return ReferenceScaling(
            ScalingSource.COMBAT_COUNT,
            Scope.TURN,
            "HAND_CARD_TYPE:Skill",
        )

    if (
        "PileType.Hand" in snippet
        and ".Cards.Count" in snippet
        and ".Cards.Count(" not in snippet
    ):
        return ReferenceScaling(
            ScalingSource.COMBAT_COUNT,
            Scope.TURN,
            "HAND_CARD_COUNT",
        )

    return ReferenceScaling(
        ScalingSource.UNPARSED,
        Scope.COMBAT,
        snippet,
    )


def _effect_family_values(
    row: Mapping[str, str],
    *,
    buffs: list[tuple[str, float]],
    debuffs: list[tuple[str, float]],
    other_powers: list[tuple[str, float]],
) -> dict[str, float]:
    """Keep unlike effect families separate instead of adding unlike units together."""
    other_known = float(len(other_powers))
    other_known += float(len(_power_entries(row.get("opaque_values"))))
    other_known += float(len(_split_list(row.get("card_tags"))))
    other_known += float(len(_split_list(row.get("tribal_references"))))
    other_known += 1.0 if _parse_bool(row.get("exhaust_pile_reference")) else 0.0
    other_known += 1.0 if _parse_float(row.get("hp_loss")) != 0 else 0.0
    if _parse_bool(row.get("applies_power_state")) and not buffs and not debuffs and not other_powers:
        other_known += 1.0

    generated_known, generated_uncertain = _count_entries(row.get("card_generation"))
    pool_known, pool_uncertain = _count_entries(row.get("pool_generation"))
    transform_known, transform_uncertain = _count_entries(row.get("card_transform"))
    tutor_known, tutor_uncertain = _count_entries(row.get("card_tutor"))
    orb_known, orb_uncertain = _count_entries(row.get("orbs_generated"))

    return {
        "heal_total": _parse_float(row.get("heal")),
        "max_hp_gain_total": _parse_float(row.get("max_hp")),
        "gold_gain_total": _parse_float(row.get("gold")),
        "card_generation_count": generated_known + pool_known,
        "card_generation_uncertain_count": generated_uncertain + pool_uncertain,
        "card_transform_count": transform_known,
        "card_transform_uncertain_count": transform_uncertain,
        "card_tutor_count": tutor_known,
        "card_tutor_uncertain_count": tutor_uncertain,
        "buff_apply_total": sum(abs(value) for _, value in buffs),
        "debuff_apply_total": sum(abs(value) for _, value in debuffs),
        "orb_generation_count": orb_known,
        "orb_generation_uncertain_count": orb_uncertain,
        "orb_evoke_count": _parse_float(row.get("orb_evoked")),
        "character_resource_gain": (
            _parse_float(row.get("forge"))
            + _parse_float(row.get("stars"))
            + _parse_float(row.get("summon"))
        ),
        "other_known_effect_count": other_known,
        "unparsed_effect_count": 1.0 if row.get("status") == "unparsed_residue" else 0.0,
    }


def _dynamic_scalings(row: Mapping[str, str]) -> tuple[ReferenceScaling, ...]:
    scalings: list[ReferenceScaling] = []
    for label in _split_list(row.get("scales_with_self_power")):
        scalings.append(
            ReferenceScaling(ScalingSource.PLAYER_VALUE, Scope.COMBAT, label, referent="self")
        )
    for label in _split_list(row.get("scales_with_enemy_power")):
        scalings.append(
            ReferenceScaling(ScalingSource.PLAYER_VALUE, Scope.COMBAT, label, referent="enemy")
        )
    if _parse_bool(row.get("exhaust_pile_reference")):
        scalings.append(ReferenceScaling(ScalingSource.COMBAT_COUNT, Scope.COMBAT, "EXHAUST"))
    if _parse_bool(row.get("discard_pile_reference")):
        scalings.append(ReferenceScaling(ScalingSource.COMBAT_COUNT, Scope.COMBAT, "DISCARD"))
    for tag in _split_list(row.get("tribal_references")):
        scalings.append(ReferenceScaling(ScalingSource.DECK_COUNT, Scope.FLOOR, tag))
    if row.get("status") == "unparsed_residue":
        scalings.append(
            ReferenceScaling(
                ScalingSource.UNPARSED,
                Scope.COMBAT,
                row.get("review_snippet") or "",
            )
        )
    return tuple(scalings)


def _parse_float(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _parse_bool(value: str | None) -> bool:
    return (value or "").strip().upper() == _TRUE


def _split_list(value: str | None) -> list[str]:
    return [part for part in (value or "").split(";") if part]


def _power_entries(value: str | None) -> list[tuple[str, float]]:
    entries: list[tuple[str, float]] = []
    for part in _split_list(value):
        label, _, rest = part.partition(":")
        magnitude_text = rest.split("(", 1)[0]
        entries.append((label, _parse_float(magnitude_text)))
    return entries


def _count_entries(value: str | None) -> tuple[float, float]:
    """Return (known total, number of entries whose count is uncertain).

    Upstream writes values such as ``SOUL:1?`` when it detected a dynamic
    repetition source but could not determine its bound. The numeric part in
    that case is a placeholder, not a trustworthy lower bound, so it must not
    be folded into a confirmed count.
    """
    known_total = 0.0
    uncertain_entries = 0.0
    for part in _split_list(value):
        _, _, rest = part.rpartition(":")
        if rest.endswith("?"):
            uncertain_entries += 1.0
            continue
        known_total += _parse_float(rest)
    return known_total, uncertain_entries
