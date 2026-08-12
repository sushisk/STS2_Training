"""Card point scoring: turns `tools/output/card_secondary_features.csv` (per-
card secondary features) into a single per-cost, 0-1 normalized card score.

This is a first-pass, HAND-WEIGHTED formula, not the final win-probability
model. Per the agreed plan, the actual deck/board evaluator will be a model
whose coefficients are FIT against real game outcomes (human run data / self-
play), not hand-picked - this script exists to produce a usable card score
now (so deck evaluation has something to sum over today) and to give that
later fitting step a sane, inspectable starting feature set. Every weight
below is a placeholder - see `_WEIGHTS` - expect to replace this whole
formula, not just retune it.

Two passes, because card generation/transform ("this card creates/replaces
itself with an unknown card from pool X") can't be priced without knowing the
average value of a card from that pool - which is itself an output of this
same scoring pass:

    Pass 1: score every card whose value doesn't depend on another card's
            score (i.e. everything except card_generation/card_transform).
    Pass 2: price each generation/transform entry at
            count * average(Pass-1 score of cards in the matching card_type
            pool, or the overall average if no pool tag/match), add it in,
            and re-normalize.

Usage:
    python tools/score_cards.py
    python tools/score_cards.py --csv tools/output/card_secondary_features.csv --output tools/output/card_scores.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Placeholder weights - a first, inspectable guess, not a calibrated model.
# Card draw and energy generation are weighted well above raw attack/defense
# because card/tempo advantage is generally understood to be strong in
# deckbuilders, but this is exactly the kind of relative weighting the later
# regression-fit pass should correct, not something to trust as final.
_WEIGHTS: dict[str, float] = {
    "effective_damage": 1.0,
    "block": 1.0,
    "cards_drawn": 3.0,
    "energy": 3.0,
    "heal": 1.0,
    "buff_magnitude": 1.5,
    "debuff_magnitude": 1.5,
    "orb_channel": 2.0,
    # Character-specific resources (Regent's Forge/Stars, Necrobinder's Osty
    # summon) - weighted well below the generic dimensions above simply
    # because there's no cross-resource common unit to compare them against
    # yet; treat these as even more provisional than the rest of this table.
    "forge": 0.2,
    "stars": 1.0,
    "summon": 1.0,
    "gold": 0.5,
    "max_hp": 2.0,
    "hp_loss": -1.0,
}
# Applied to a card flagged applies_power_state==TRUE with NO buff/debuff
# magnitude captured (i.e. a wholly unclassified unique Power) - since its
# true value is unknown, this uses rarity as the only available signal
# rather than guessing a number outright.
_UNCLASSIFIED_POWER_BONUS_BY_RARITY: dict[str, float] = {
    "Basic": 1.0,
    "Common": 1.5,
    "Uncommon": 2.5,
    "Rare": 4.0,
    "Ancient": 4.0,
    "Event": 2.5,
}
_DEFAULT_UNCLASSIFIED_POWER_BONUS = 2.0
# A fixed downside for generating a card that's itself excluded as
# Curse/Status-equivalent (deck clutter, e.g. FightThrough -> Wound) - a
# placeholder, same spirit as every other weight in this module.
_CURSE_EQUIVALENT_GENERATION_PENALTY = -2.0


@dataclass
class CardScore:
    card_id: str
    card_type: str
    rarity: str
    energy_cost: float | None
    effective_cost: float | None
    raw_value: float | None
    per_cost_value: float | None
    score: float | None = None
    unscored_reason: str | None = None
    generation_pool_tags: list[str] = field(default_factory=list)
    transform_pool_tags: list[str] = field(default_factory=list)
    generation_count: float = 0.0
    transform_count: float = 0.0


def _parse_float(value: str) -> float:
    return float(value) if value else 0.0


def _parse_power_list(value: str) -> list[tuple[str, float]]:
    """`"WEAK_POWER:2(Counter);VULNERABLE_POWER:1(Counter)"` -> [("WEAK_POWER",
    2.0), ("VULNERABLE_POWER", 1.0)] - stack_type suffix, if any, is dropped."""
    entries = []
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        label, _, rest = part.partition(":")
        magnitude_text = rest.split("(", 1)[0]
        try:
            entries.append((label, float(magnitude_text)))
        except ValueError:
            continue
    return entries


def _parse_created_list(value: str) -> list[tuple[str, float, bool]]:
    """`"WOUND:2;SHIV:1?"` -> [("WOUND", 2.0, False), ("SHIV", 1.0, True)]."""
    entries = []
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        label, _, rest = part.partition(":")
        uncertain = rest.endswith("?")
        count_text = rest[:-1] if uncertain else rest
        try:
            entries.append((label, float(count_text), uncertain))
        except ValueError:
            continue
    return entries


def _parse_pool_list(value: str) -> list[tuple[str, float, bool]]:
    """`"type:Power:1;unfiltered:1?"` -> [("type:Power", 1.0, False),
    ("unfiltered", 1.0, True)] - the pool_filter tag itself may contain a
    colon, so this splits on the LAST colon rather than the first."""
    entries = []
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        label, _, rest = part.rpartition(":")
        if not label:
            continue
        uncertain = rest.endswith("?")
        count_text = rest[:-1] if uncertain else rest
        try:
            entries.append((label, float(count_text), uncertain))
        except ValueError:
            continue
    return entries


def _effective_cost(energy_cost: float, *, is_x_cost: bool) -> float | None:
    if is_x_cost or energy_cost < 0:
        return None  # X-cost / non-normalizable - see module docstring's Pass split
    if energy_cost == 0:
        return 0.5
    return energy_cost


def _base_raw_value(row: dict[str, str]) -> float:
    damage = _parse_float(row["damage"]) + _parse_float(row["extra_damage"])
    repeat_hits = _parse_float(row["repeat_hits"])
    effective_damage = damage * (1 + repeat_hits) if damage else 0.0

    buff_magnitude = sum(v for _, v in _parse_power_list(row["buffs_applied"]))
    debuff_magnitude = sum(v for _, v in _parse_power_list(row["debuffs_applied"]))

    orb_count = sum(v for _, v, _ in _parse_created_list(row["orbs_generated"]))

    value = 0.0
    value += _WEIGHTS["effective_damage"] * effective_damage
    value += _WEIGHTS["block"] * _parse_float(row["block"])
    value += _WEIGHTS["cards_drawn"] * _parse_float(row["cards_drawn"])
    value += _WEIGHTS["energy"] * _parse_float(row["energy"])
    value += _WEIGHTS["heal"] * _parse_float(row["heal"])
    value += _WEIGHTS["buff_magnitude"] * buff_magnitude
    value += _WEIGHTS["debuff_magnitude"] * debuff_magnitude
    value += _WEIGHTS["orb_channel"] * orb_count
    value += _WEIGHTS["forge"] * _parse_float(row["forge"])
    value += _WEIGHTS["stars"] * _parse_float(row["stars"])
    value += _WEIGHTS["summon"] * _parse_float(row["summon"])
    value += _WEIGHTS["gold"] * _parse_float(row["gold"])
    value += _WEIGHTS["max_hp"] * _parse_float(row["max_hp"])
    value += _WEIGHTS["hp_loss"] * _parse_float(row["hp_loss"])

    if row["applies_power_state"] == "TRUE" and buff_magnitude == 0.0 and debuff_magnitude == 0.0:
        value += _UNCLASSIFIED_POWER_BONUS_BY_RARITY.get(
            row["rarity"], _DEFAULT_UNCLASSIFIED_POWER_BONUS
        )

    return value


def score_cards(rows: list[dict[str, str]]) -> list[CardScore]:
    pass1: list[CardScore] = []
    for row in rows:
        if row["deck_eval_excluded"] == "TRUE":
            pass1.append(
                CardScore(
                    card_id=row["card_id"],
                    card_type=row["card_type"],
                    rarity=row["rarity"],
                    energy_cost=_parse_float(row["energy_cost"]),
                    effective_cost=None,
                    raw_value=None,
                    per_cost_value=None,
                    unscored_reason=f"deck_eval_excluded:{row.get('exclusion_reason') or 'unknown'}",
                )
            )
            continue

        cost = _parse_float(row["energy_cost"])
        effective_cost = _effective_cost(cost, is_x_cost=row.get("is_x_cost") == "TRUE")
        generation = _parse_created_list(row["card_generation"])
        transform = _parse_created_list(row["card_transform"])

        card = CardScore(
            card_id=row["card_id"],
            card_type=row["card_type"],
            rarity=row["rarity"],
            energy_cost=cost,
            effective_cost=effective_cost,
            raw_value=None,
            per_cost_value=None,
            generation_count=sum(c for _, c, _ in generation),
            transform_count=sum(c for _, c, _ in transform),
        )
        if effective_cost is None:
            card.unscored_reason = "x_cost_or_negative_cost"
            pass1.append(card)
            continue

        raw_value = _base_raw_value(row)
        card.raw_value = raw_value  # provisional - generation/transform priced in pass 2
        card.per_cost_value = raw_value / effective_cost
        pass1.append(card)

    # Pass 2: price generation/transform/pool-generation/tutor entries.
    #
    # Known target (card_generation/card_transform - CreateCard<T>() names the
    # exact class): use THAT card's own pass-1 score if we have one. If the
    # target is itself Curse/Status (e.g. FightThrough/CrashLanding generating
    # Wound/Debris - a downside, not a reward), it has no pass-1 score at all
    # (excluded from normal scoring) - price it as a fixed penalty instead of
    # silently falling back to a positive average.
    #
    # Unknown target (pool_generation/card_tutor - RNG/player choice from a
    # filtered pool): use the average score of cards sharing the pool_filter's
    # `type:X` tag, falling back to the overall average when the filter is
    # unfiltered or doesn't match a known card_type bucket.
    scored_pass1 = [c for c in pass1 if c.per_cost_value is not None]
    overall_avg = statistics.mean(c.per_cost_value for c in scored_pass1) if scored_pass1 else 0.0
    avg_by_type: dict[str, float] = {}
    for card_type in {c.card_type for c in scored_pass1}:
        values = [c.per_cost_value for c in scored_pass1 if c.card_type == card_type]
        avg_by_type[card_type] = statistics.mean(values) if values else overall_avg
    score_by_id = {c.card_id: c.per_cost_value for c in scored_pass1}

    rows_by_id = {row["card_id"]: row for row in rows}
    for card in pass1:
        if card.effective_cost is None:
            continue
        row = rows_by_id[card.card_id]
        bonus = 0.0

        # Known target card (its own class name is in the entry): price at
        # its own pass-1 score, or the fixed penalty if it's a Curse/Status-
        # equivalent downside card, falling back to the overall average only
        # if the id doesn't resolve to any row at all.
        for created_id, count, _uncertain in (
            *_parse_created_list(row["card_generation"]),
            *_parse_created_list(row["card_transform"]),
        ):
            target_row = rows_by_id.get(created_id)
            if created_id in score_by_id:
                per_unit = score_by_id[created_id]
            elif target_row is not None and target_row.get("deck_eval_excluded") == "TRUE":
                per_unit = _CURSE_EQUIVALENT_GENERATION_PENALTY
            else:
                per_unit = overall_avg
            bonus += count * per_unit

        # Unknown target (RNG/player choice from a filtered pool): price at
        # the average score of cards sharing the pool_filter's `type:X` tag.
        for pool_filter, count, _uncertain in (
            *_parse_pool_list(row["pool_generation"]),
            *_parse_pool_list(row["card_tutor"]),
        ):
            card_type_tag = pool_filter.removeprefix("type:")
            per_unit = avg_by_type.get(card_type_tag, overall_avg)
            bonus += count * per_unit

        if bonus:
            card.raw_value += bonus
            card.per_cost_value = card.raw_value / card.effective_cost

    # Final 0-1 normalization: min-max AFTER clipping to [p1, p99]. Plain
    # min-max was tried first and turned out unusable - a single genuinely
    # extreme card (e.g. GRAND_FINALE: 60 damage to all enemies at 0 cost,
    # albeit only playable with an empty draw pile - see the module
    # docstring's note on conditional-playability not being modeled) stretches
    # the whole scale so far that almost every other card collapses to ~0.
    # Clipping the top/bottom 1% keeps the bulk of cards spread out over the
    # full 0-1 range and pins outliers at 0/1 instead of letting them dominate.
    scored = [c for c in pass1 if c.per_cost_value is not None]
    if scored:
        values = sorted(c.per_cost_value for c in scored)
        lo = _percentile(values, 0.01)
        hi = _percentile(values, 0.99)
        span = (hi - lo) or 1.0
        for card in scored:
            clipped = min(max(card.per_cost_value, lo), hi)
            card.score = (clipped - lo) / span

    return pass1


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=Path("tools/output/card_secondary_features.csv"))
    parser.add_argument("--output", type=Path, default=Path("tools/output/card_scores.csv"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--top", type=int, default=15, help="how many top/bottom cards to print")
    args = parser.parse_args()

    with args.csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    results = score_cards(rows)
    scored = sorted((c for c in results if c.score is not None), key=lambda c: c.score, reverse=True)
    unscored = [c for c in results if c.score is None]

    unscored_reason_counts = Counter(c.unscored_reason for c in unscored)
    print(f"Scored {len(scored)}/{len(results)} cards (0-1, per-cost, min-max normalized).")
    print(f"Unscored: {len(unscored)} {dict(unscored_reason_counts)}")

    print(f"\nTop {args.top} cards:")
    for card in scored[: args.top]:
        print(f"  {card.score:.3f}  {card.card_id:22s} cost={card.energy_cost:>4g} {card.card_type}")

    print(f"\nBottom {args.top} cards:")
    for card in scored[-args.top :]:
        print(f"  {card.score:.3f}  {card.card_id:22s} cost={card.energy_cost:>4g} {card.card_type}")

    if args.output.exists() and not args.force:
        print(f"\n{args.output} already exists - pass --force to overwrite.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["card_id", "card_type", "rarity", "energy_cost", "effective_cost", "raw_value", "per_cost_value", "score", "unscored_reason"])
        for card in results:
            writer.writerow(
                [
                    card.card_id,
                    card.card_type,
                    card.rarity,
                    card.energy_cost,
                    card.effective_cost,
                    card.raw_value,
                    card.per_cost_value,
                    card.score,
                    card.unscored_reason or "",
                ]
            )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
