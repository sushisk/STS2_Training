from __future__ import annotations

import pytest

from sts2_training.board_eval.card_features import CardFeatures, ReferenceScaling, ScalingSource, Scope
from sts2_training.board_eval.deck_summary import DECK_SUMMARY_FEATURE_NAMES, DeckSummary, summarize_deck


def _attack(**overrides: object) -> CardFeatures:
    values: dict[str, object] = {
        "card_id": "A",
        "is_attack": True,
        "cost": 1.0,
        "damage": 6.0,
    }
    values.update(overrides)
    return CardFeatures(**values)  # type: ignore[arg-type]


def _skill(**overrides: object) -> CardFeatures:
    values: dict[str, object] = {
        "card_id": "S",
        "is_skill": True,
        "cost": 1.0,
        "block": 5.0,
    }
    values.update(overrides)
    return CardFeatures(**values)  # type: ignore[arg-type]


def test_empty_deck_is_zeroed() -> None:
    assert summarize_deck([]) == DeckSummary()


def test_type_cost_damage_and_block_aggregates() -> None:
    summary = summarize_deck([_attack(), _attack(cost=2.0, damage=10.0), _skill(block=7.0)])

    assert summary.deck_size == 3
    assert summary.attack_count == 2
    assert summary.skill_count == 1
    assert summary.attack_ratio == pytest.approx(2 / 3)
    assert summary.cost_1_count == 2
    assert summary.cost_2_count == 1
    assert summary.total_base_damage == 16.0
    assert summary.max_damage == 10.0
    assert summary.total_block == 7.0


def test_damage_and_block_efficiency_use_only_effect_contributors() -> None:
    summary = summarize_deck([_attack(cost=1.0, damage=6.0), _skill(cost=1.0, block=5.0)])

    assert summary.damage_per_energy == pytest.approx(6.0)
    assert summary.block_per_energy == pytest.approx(5.0)


def test_star_cost_cards_are_separate_from_energy_cost_features() -> None:
    star_attack = _attack(cost=0.0, star_cost=3.0, damage=9.0)
    energy_attack = _attack(cost=1.0, damage=6.0)

    summary = summarize_deck([star_attack, energy_attack])

    assert summary.cost_0_count == 0
    assert summary.cost_1_count == 1
    assert summary.star_cost_card_count == 1
    assert summary.total_star_cost == 3.0
    assert summary.star_cost_1_count == 0
    assert summary.star_cost_2_count == 0
    assert summary.star_cost_3plus_count == 1
    assert summary.total_base_damage == 15.0
    # The Star-cost card contributes to total damage, but is not treated as
    # zero-Energy damage when computing Energy efficiency.
    assert summary.damage_per_energy == pytest.approx(6.0)


def test_upgrade_counts_are_final_upgrade_representation() -> None:
    summary = summarize_deck(
        [
            _attack(upgrade=True),
            _attack(),
            _skill(upgrade=True),
            CardFeatures(card_id="P", is_power=True, upgrade=True),
        ]
    )

    assert summary.upgraded_count == 3
    assert summary.upgrade_ratio == pytest.approx(0.75)
    assert summary.upgraded_attack_count == 1
    assert summary.upgraded_skill_count == 1
    assert summary.upgraded_power_count == 1


def test_unknown_cards_contribute_to_deck_size_and_coverage() -> None:
    summary = summarize_deck([_attack(), _skill()], unknown_card_count=2)

    assert summary.deck_size == 4
    assert summary.unknown_card_count == 2
    assert summary.known_card_ratio == pytest.approx(0.5)
    assert summary.attack_ratio == pytest.approx(0.25)


def test_excluded_and_unplayable_cards_do_not_pollute_mechanical_aggregates() -> None:
    curse = CardFeatures(
        card_id="CURSE",
        card_type="Curse",
        keywords=("Unplayable",),
        deck_eval_excluded=True,
        cost=-1.0,
        damage=40.0,
        block=30.0,
        draw=3.0,
        energy_gain=2.0,
        heal_total=9.0,
    )
    status = CardFeatures(
        card_id="STATUS",
        card_type="Status",
        cost=-1.0,
        damage=20.0,
        block=15.0,
    )
    unplayable_quest = CardFeatures(
        card_id="QUEST",
        card_type="Quest",
        keywords=("Unplayable",),
        cost=-1.0,
        damage=50.0,
    )

    summary = summarize_deck([_attack(cost=2.0, damage=8.0), curse, status, unplayable_quest])

    assert summary.deck_size == 4
    assert summary.attack_count == 1
    assert summary.attack_ratio == pytest.approx(0.25)
    assert summary.curse_count == 1
    assert summary.status_count == 1
    assert summary.unplayable_count == 2
    assert summary.excluded_card_count == 3
    assert summary.cost_2_count == 1
    assert summary.total_base_damage == 8.0
    assert summary.damage_per_energy == pytest.approx(4.0)
    assert summary.max_damage == 8.0
    assert summary.total_block == 0.0
    assert summary.draw_amount == 0.0
    assert summary.energy_generation == 0.0
    assert summary.heal_total == 0.0


def test_effect_family_totals_remain_separate() -> None:
    summary = summarize_deck(
        [
            _skill(heal_total=2.0, gold_gain_total=5.0, buff_apply_total=1.0),
            _skill(heal_total=3.0, gold_gain_total=7.0, debuff_apply_total=4.0),
        ]
    )

    assert summary.heal_total == 5.0
    assert summary.gold_gain_total == 12.0
    assert summary.buff_apply_total == 1.0
    assert summary.debuff_apply_total == 4.0


def test_uncertain_effect_counts_stay_separate_at_deck_level() -> None:
    summary = summarize_deck(
        [
            _skill(
                card_generation_count=2.0,
                card_generation_uncertain_count=1.0,
                orb_generation_count=1.0,
                orb_generation_uncertain_count=2.0,
            ),
            _skill(
                card_generation_uncertain_count=1.0,
                card_transform_uncertain_count=1.0,
            ),
        ]
    )

    assert summary.card_generation_count == 2.0
    assert summary.card_generation_uncertain_count == 2.0
    assert summary.card_transform_count == 0.0
    assert summary.card_transform_uncertain_count == 1.0
    assert summary.orb_generation_count == 1.0
    assert summary.orb_generation_uncertain_count == 2.0


def test_dynamic_payoff_uses_filter_value() -> None:
    exhaust = CardFeatures(
        card_id="E",
        dynamic_scalings=(ReferenceScaling(ScalingSource.COMBAT_COUNT, Scope.COMBAT, "EXHAUST"),),
    )
    discard = CardFeatures(
        card_id="D",
        dynamic_scalings=(ReferenceScaling(ScalingSource.COMBAT_COUNT, Scope.COMBAT, "DISCARD"),),
    )

    summary = summarize_deck([exhaust, discard])

    assert summary.exhaust_payoff_count == 1
    assert summary.discard_payoff_count == 1
    assert summary.dynamic_card_count == 2


def test_vector_contract_matches_dataclass_schema() -> None:
    summary = summarize_deck([_attack()])

    assert DECK_SUMMARY_FEATURE_NAMES == tuple(DeckSummary.__dataclass_fields__)
    assert len(summary.to_vector()) == len(DECK_SUMMARY_FEATURE_NAMES)


def test_negative_unknown_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        summarize_deck([], unknown_card_count=-1)
