"""Deck-level aggregate features for the Run-state board evaluation model."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from sts2_training.board_eval.card_features import CardFeatures, ScalingSource


@dataclass(frozen=True)
class DeckSummary:
    deck_size: int = 0
    attack_count: int = 0
    skill_count: int = 0
    power_count: int = 0
    attack_ratio: float = 0.0
    skill_ratio: float = 0.0
    power_ratio: float = 0.0

    upgraded_count: int = 0
    upgrade_ratio: float = 0.0
    upgraded_attack_count: int = 0
    upgraded_skill_count: int = 0
    upgraded_power_count: int = 0
    unknown_card_count: int = 0
    known_card_ratio: float = 0.0
    curse_count: int = 0
    status_count: int = 0
    unplayable_count: int = 0
    excluded_card_count: int = 0

    cost_0_count: int = 0
    cost_1_count: int = 0
    cost_2_count: int = 0
    cost_3plus_count: int = 0
    cost_x_count: int = 0
    star_cost_card_count: int = 0
    total_star_cost: float = 0.0
    star_cost_1_count: int = 0
    star_cost_2_count: int = 0
    star_cost_3plus_count: int = 0

    total_base_damage: float = 0.0
    damage_per_energy: float = 0.0
    max_damage: float = 0.0
    total_block: float = 0.0
    block_per_energy: float = 0.0
    max_block: float = 0.0
    draw_amount: float = 0.0
    energy_generation: float = 0.0
    aoe_count: int = 0
    multi_hit_count: int = 0
    exhaust_enable_count: int = 0
    exhaust_payoff_count: int = 0
    discard_enable_count: int = 0
    discard_payoff_count: int = 0
    poison_count: int = 0

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
    dynamic_card_count: int = 0

    def to_vector(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in DECK_SUMMARY_FEATURE_NAMES)


DECK_SUMMARY_FEATURE_NAMES: tuple[str, ...] = tuple(DeckSummary.__dataclass_fields__)


def summarize_deck(
    cards: Iterable[CardFeatures],
    *,
    unknown_card_count: int = 0,
) -> DeckSummary:
    cards = list(cards)
    if unknown_card_count < 0:
        raise ValueError("unknown_card_count must be non-negative")
    known_count = len(cards)
    deck_size = known_count + unknown_card_count
    if deck_size == 0:
        return DeckSummary()

    # The extraction catalog marks Curse/Status and other unsupported cards as
    # deck-eval-excluded, and Unplayable can also appear without that catalog
    # flag (for example a Quest card). They still occupy deck slots, but their
    # damage/cost/etc. must not be interpreted as normal playable-card value.
    mechanical_cards = [card for card in cards if not _is_mechanically_excluded(card)]
    energy_cost_cards = [
        card for card in mechanical_cards if not card.is_x_cost and not card.uses_star_cost
    ]
    star_cost_cards = [card for card in mechanical_cards if card.uses_star_cost]

    attack_count = sum(1 for card in mechanical_cards if card.is_attack)
    skill_count = sum(1 for card in mechanical_cards if card.is_skill)
    power_count = sum(1 for card in mechanical_cards if card.is_power)
    upgraded_count = sum(1 for card in cards if card.upgrade)
    total_base_damage = sum(card.damage for card in mechanical_cards)
    total_block = sum(card.block for card in mechanical_cards)
    damage_per_energy = _effect_per_energy(energy_cost_cards, effect_name="damage")
    block_per_energy = _effect_per_energy(energy_cost_cards, effect_name="block")
    exhaust_payoff, discard_payoff = _dynamic_scaling_filter_counts(mechanical_cards)

    return DeckSummary(
        deck_size=deck_size,
        attack_count=attack_count,
        skill_count=skill_count,
        power_count=power_count,
        attack_ratio=attack_count / deck_size,
        skill_ratio=skill_count / deck_size,
        power_ratio=power_count / deck_size,
        upgraded_count=upgraded_count,
        upgrade_ratio=upgraded_count / deck_size,
        upgraded_attack_count=sum(
            1 for card in mechanical_cards if card.upgrade and card.is_attack
        ),
        upgraded_skill_count=sum(
            1 for card in mechanical_cards if card.upgrade and card.is_skill
        ),
        upgraded_power_count=sum(
            1 for card in mechanical_cards if card.upgrade and card.is_power
        ),
        unknown_card_count=unknown_card_count,
        known_card_ratio=known_count / deck_size,
        curse_count=sum(1 for card in cards if card.card_type == "Curse"),
        status_count=sum(1 for card in cards if card.card_type == "Status"),
        unplayable_count=sum(1 for card in cards if "Unplayable" in card.keywords),
        excluded_card_count=sum(1 for card in cards if _is_mechanically_excluded(card)),
        cost_0_count=sum(1 for card in energy_cost_cards if card.cost == 0),
        cost_1_count=sum(1 for card in energy_cost_cards if card.cost == 1),
        cost_2_count=sum(1 for card in energy_cost_cards if card.cost == 2),
        cost_3plus_count=sum(1 for card in energy_cost_cards if card.cost >= 3),
        cost_x_count=sum(1 for card in mechanical_cards if card.is_x_cost),
        star_cost_card_count=len(star_cost_cards),
        total_star_cost=sum(card.star_cost for card in star_cost_cards),
        star_cost_1_count=sum(1 for card in star_cost_cards if card.star_cost == 1),
        star_cost_2_count=sum(1 for card in star_cost_cards if card.star_cost == 2),
        star_cost_3plus_count=sum(1 for card in star_cost_cards if card.star_cost >= 3),
        total_base_damage=total_base_damage,
        damage_per_energy=damage_per_energy,
        max_damage=max((card.damage for card in mechanical_cards), default=0.0),
        total_block=total_block,
        block_per_energy=block_per_energy,
        max_block=max((card.block for card in mechanical_cards), default=0.0),
        draw_amount=sum(card.draw for card in mechanical_cards),
        energy_generation=sum(card.energy_gain for card in mechanical_cards),
        aoe_count=sum(1 for card in mechanical_cards if card.is_aoe),
        multi_hit_count=sum(1 for card in mechanical_cards if card.is_multi_hit),
        exhaust_enable_count=sum(
            1 for card in mechanical_cards if card.exhaust or card.exhaust_generation
        ),
        exhaust_payoff_count=exhaust_payoff,
        discard_enable_count=sum(1 for card in mechanical_cards if card.discard),
        discard_payoff_count=discard_payoff,
        poison_count=sum(1 for card in mechanical_cards if card.poison),
        heal_total=sum(card.heal_total for card in mechanical_cards),
        max_hp_gain_total=sum(card.max_hp_gain_total for card in mechanical_cards),
        gold_gain_total=sum(card.gold_gain_total for card in mechanical_cards),
        card_generation_count=sum(card.card_generation_count for card in mechanical_cards),
        card_generation_uncertain_count=sum(
            card.card_generation_uncertain_count for card in mechanical_cards
        ),
        card_transform_count=sum(card.card_transform_count for card in mechanical_cards),
        card_transform_uncertain_count=sum(
            card.card_transform_uncertain_count for card in mechanical_cards
        ),
        card_tutor_count=sum(card.card_tutor_count for card in mechanical_cards),
        card_tutor_uncertain_count=sum(
            card.card_tutor_uncertain_count for card in mechanical_cards
        ),
        buff_apply_total=sum(card.buff_apply_total for card in mechanical_cards),
        debuff_apply_total=sum(card.debuff_apply_total for card in mechanical_cards),
        orb_generation_count=sum(card.orb_generation_count for card in mechanical_cards),
        orb_generation_uncertain_count=sum(
            card.orb_generation_uncertain_count for card in mechanical_cards
        ),
        orb_evoke_count=sum(card.orb_evoke_count for card in mechanical_cards),
        character_resource_gain=sum(card.character_resource_gain for card in mechanical_cards),
        other_known_effect_count=sum(card.other_known_effect_count for card in mechanical_cards),
        unparsed_effect_count=sum(card.unparsed_effect_count for card in mechanical_cards),
        dynamic_card_count=sum(1 for card in mechanical_cards if card.is_dynamic),
    )


def _effect_per_energy(cards: Sequence[CardFeatures], *, effect_name: str) -> float:
    contributors = [card for card in cards if float(getattr(card, effect_name)) > 0]
    total_cost = sum(card.cost for card in contributors if card.cost >= 0)
    if total_cost <= 0:
        return 0.0
    return sum(float(getattr(card, effect_name)) for card in contributors) / total_cost


def _is_mechanically_excluded(card: CardFeatures) -> bool:
    return (
        card.deck_eval_excluded
        or card.card_type in {"Curse", "Status"}
        or "Unplayable" in card.keywords
    )


def _dynamic_scaling_filter_counts(cards: Sequence[CardFeatures]) -> tuple[int, int]:
    exhaust_payoff = 0
    discard_payoff = 0
    for card in cards:
        for scaling in card.dynamic_scalings:
            if scaling.source is not ScalingSource.COMBAT_COUNT:
                continue
            if scaling.filter_value == "EXHAUST":
                exhaust_payoff += 1
            elif scaling.filter_value == "DISCARD":
                discard_payoff += 1
    return exhaust_payoff, discard_payoff
