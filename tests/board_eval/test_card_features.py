from __future__ import annotations

import pytest

from sts2_training.board_eval.card_features import (
    DEFAULT_CARD_FEATURES_CSV,
    FEATURE_NAMES,
    CardFeatureExtractor,
    CardFeatures,
    ScalingSource,
    Scope,
    UnknownCardError,
)


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "card_id": "CARD",
        "card_type": "Skill",
        "target_type": "Self",
        "energy_cost": "1",
    }
    row.update(overrides)
    return row


def test_basic_attack_maps_cost_damage_and_scaling() -> None:
    extractor = CardFeatureExtractor(
        [_row(card_id="STRIKE", card_type="Attack", target_type="AnyEnemy", damage="6")]
    )

    features = extractor.extract("STRIKE")

    assert features.cost == 1.0
    assert features.damage == 6.0
    assert features.is_attack is True
    assert features.strength_scaling is True


def test_star_cost_is_retained_separately_from_energy_cost() -> None:
    extractor = CardFeatureExtractor(
        [_row(card_id="STAR_ATTACK", card_type="Attack", energy_cost="0", star_cost="3", damage="6")]
    )

    features = extractor.extract("STAR_ATTACK")

    assert features.cost == 0.0
    assert features.star_cost == 3.0
    assert features.uses_star_cost is True
    vector = features.to_vector()
    assert vector[FEATURE_NAMES.index("star_cost")] == 3.0
    assert vector[FEATURE_NAMES.index("uses_star_cost")] == 1.0


def test_upgrade_remains_instance_count_signal_only() -> None:
    extractor = CardFeatureExtractor([_row(card_id="STRIKE", damage="6")])

    base = extractor.extract("STRIKE")
    upgraded = extractor.extract("STRIKE", upgraded=True)

    assert base.upgrade is False
    assert upgraded.upgrade is True
    assert upgraded.damage == base.damage


def test_effect_families_are_separate_features() -> None:
    extractor = CardFeatureExtractor(
        [
            _row(
                card_id="MIXED",
                heal="6",
                max_hp="2",
                gold="10",
                card_generation="SHIV:2",
                pool_generation="ATTACK_POOL:1",
                card_transform="RANDOM:1",
                card_tutor="Strike:1",
                buffs_applied="STRENGTH_POWER:2(Counter)",
                debuffs_applied="WEAK_POWER:3(Counter)",
                orbs_generated="FROST:2",
                orb_evoked="1",
                forge="2",
                stars="3",
                summon="4",
            )
        ]
    )

    features = extractor.extract("MIXED")

    assert features.heal_total == 6.0
    assert features.max_hp_gain_total == 2.0
    assert features.gold_gain_total == 10.0
    assert features.card_generation_count == 3.0
    assert features.card_transform_count == 1.0
    assert features.card_tutor_count == 1.0
    assert features.buff_apply_total == 2.0
    assert features.debuff_apply_total == 3.0
    assert features.orb_generation_count == 2.0
    assert features.orb_evoke_count == 1.0
    assert features.character_resource_gain == 9.0
    assert "other_effect_magnitude" not in FEATURE_NAMES


def test_uncertain_effect_counts_are_not_promoted_to_confirmed_counts() -> None:
    extractor = CardFeatureExtractor(
        [
            _row(
                card_id="UNCERTAIN",
                card_generation="SOUL:1?;SHIV:2",
                pool_generation="type:Attack:1?",
                card_transform="RANDOM:1?",
                card_tutor="Strike:1?;Skill:2",
                orbs_generated="FrostOrb:1?;LightningOrb:2",
            )
        ]
    )

    features = extractor.extract("UNCERTAIN")

    assert features.card_generation_count == 2.0
    assert features.card_generation_uncertain_count == 2.0
    assert features.card_transform_count == 0.0
    assert features.card_transform_uncertain_count == 1.0
    assert features.card_tutor_count == 2.0
    assert features.card_tutor_uncertain_count == 1.0
    assert features.orb_generation_count == 2.0
    assert features.orb_generation_uncertain_count == 1.0


def test_other_known_and_unparsed_counts_are_not_magnitudes() -> None:
    extractor = CardFeatureExtractor(
        [
            _row(
                card_id="ODD",
                applies_power_state="TRUE",
                hp_loss="7",
                card_tags="TagA;TagB",
                status="unparsed_residue",
                review_snippet="mystery",
            )
        ]
    )

    features = extractor.extract("ODD")

    assert features.other_known_effect_count == 4.0
    assert features.unparsed_effect_count == 1.0


def test_reference_scaling_splits_referent_and_filter_value() -> None:
    extractor = CardFeatureExtractor(
        [_row(card_id="BULLY", scales_with_enemy_power="VULNERABLE_POWER")]
    )

    scaling = extractor.extract("BULLY").dynamic_scalings[0]

    assert scaling.source is ScalingSource.PLAYER_VALUE
    assert scaling.scope is Scope.COMBAT
    assert scaling.referent == "enemy"
    assert scaling.filter_value == "VULNERABLE_POWER"
    assert not hasattr(scaling, "detail")


def test_pile_and_tribal_references_keep_structured_filter() -> None:
    extractor = CardFeatureExtractor(
        [
            _row(
                card_id="DYNAMIC",
                exhaust_pile_reference="TRUE",
                discard_pile_reference="TRUE",
                tribal_references="Strike",
            )
        ]
    )

    scalings = extractor.extract("DYNAMIC").dynamic_scalings
    filters = {(item.source, item.filter_value) for item in scalings}

    assert (ScalingSource.COMBAT_COUNT, "EXHAUST") in filters
    assert (ScalingSource.COMBAT_COUNT, "DISCARD") in filters
    assert (ScalingSource.DECK_COUNT, "Strike") in filters


def test_fixed_repeat_multi_hit_does_not_invent_reference() -> None:
    extractor = CardFeatureExtractor(
        [_row(card_id="FIXED_HITS", card_type="Attack", damage="3", repeat_hits="4")]
    )

    features = extractor.extract("FIXED_HITS")

    assert features.is_multi_hit is True
    assert features.hit_count_reference is None


def test_repeat_hits_does_not_hide_dynamic_hit_reference() -> None:
    extractor = CardFeatureExtractor(
        [
            _row(
                card_id="REPEAT_AND_DYNAMIC",
                card_type="Attack",
                damage="5",
                repeat_hits="1",
                review_snippet=(
                    'new RepeatVar(1), new CalculatedVar("CalculatedHits").WithMultiplier('
                    "(CardModel card, Creature? _) => "
                    "card.Owner.PlayerCombatState.OrbQueue.Orbs.Count)"
                ),
            )
        ]
    )

    reference = extractor.extract("REPEAT_AND_DYNAMIC").hit_count_reference

    assert reference is not None
    assert reference.source is ScalingSource.COMBAT_COUNT
    assert reference.filter_value == "ORB_COUNT"


def test_dynamic_multi_hit_retains_orb_count_reference() -> None:
    extractor = CardFeatureExtractor(
        [
            _row(
                card_id="ORB_HITS",
                card_type="Attack",
                damage="5",
                status="unparsed_residue",
                review_snippet=(
                    'new CalculatedVar("CalculatedHits").WithMultiplier('
                    "(CardModel card, Creature? _) => "
                    "card.Owner.PlayerCombatState.OrbQueue.Orbs.Count)"
                ),
            )
        ]
    )

    features = extractor.extract("ORB_HITS")
    reference = features.hit_count_reference

    assert features.is_multi_hit is True
    assert reference is not None
    assert reference.source is ScalingSource.COMBAT_COUNT
    assert reference.scope is Scope.COMBAT
    assert reference.filter_value == "ORB_COUNT"


def test_dynamic_multi_hit_retains_hand_skill_reference() -> None:
    extractor = CardFeatureExtractor(
        [
            _row(
                card_id="HAND_SKILL_HITS",
                card_type="Attack",
                damage="5",
                status="unparsed_residue",
                review_snippet=(
                    'new CalculatedVar("CalculatedHits").WithMultiplier('
                    "(CardModel card, Creature? _) => "
                    "PileType.Hand.GetPile(card.Owner).Cards.Count("
                    "(CardModel c) => c.Type == CardType.Skill))"
                ),
            )
        ]
    )

    reference = extractor.extract("HAND_SKILL_HITS").hit_count_reference

    assert reference is not None
    assert reference.source is ScalingSource.COMBAT_COUNT
    assert reference.scope is Scope.TURN
    assert reference.filter_value == "HAND_CARD_TYPE:Skill"


def test_incomplete_filtered_hand_count_is_unparsed() -> None:
    snippet = (
        'new CalculatedVar("CalculatedHits").WithMultiplier('
        "(CardModel card, Creature? _) => "
        "PileType.Hand.GetPile(card.Owner).Cards.Count((CardMo"
    )
    extractor = CardFeatureExtractor(
        [
            _row(
                card_id="TRUNCATED_FILTER",
                card_type="Attack",
                damage="5",
                review_snippet=snippet,
            )
        ]
    )

    reference = extractor.extract("TRUNCATED_FILTER").hit_count_reference

    assert reference is not None
    assert reference.source is ScalingSource.UNPARSED
    assert reference.filter_value == snippet


def test_unknown_dynamic_hit_reference_is_preserved_without_guessing() -> None:
    snippet = (
        'new CalculatedVar("CalculatedHits").WithMultiplier('
        "(CardModel card, Creature? _) => SomeFutureMechanic.Count(card))"
    )
    extractor = CardFeatureExtractor(
        [
            _row(
                card_id="UNKNOWN_HITS",
                card_type="Attack",
                damage="5",
                status="unparsed_residue",
                review_snippet=snippet,
            )
        ]
    )

    features = extractor.extract("UNKNOWN_HITS")
    reference = features.hit_count_reference

    assert features.is_multi_hit is True
    assert reference is not None
    assert reference.source is ScalingSource.UNPARSED
    assert reference.scope is Scope.COMBAT
    assert reference.filter_value == snippet


def test_unknown_card_id_raises() -> None:
    extractor = CardFeatureExtractor([_row(card_id="KNOWN")])

    with pytest.raises(UnknownCardError):
        extractor.extract("UNKNOWN")


def test_to_vector_matches_feature_schema() -> None:
    features = CardFeatures(card_id="X", is_attack=True, heal_total=2.0)

    vector = features.to_vector()

    assert len(vector) == len(FEATURE_NAMES)
    assert vector[FEATURE_NAMES.index("is_attack")] == 1.0
    assert vector[FEATURE_NAMES.index("heal_total")] == 2.0


def test_default_catalog_path_is_repo_anchored() -> None:
    assert DEFAULT_CARD_FEATURES_CSV.is_absolute()
    assert DEFAULT_CARD_FEATURES_CSV.name == "card_secondary_features.csv"


@pytest.mark.skipif(not DEFAULT_CARD_FEATURES_CSV.exists(), reason="card catalog not present")
def test_real_catalog_retains_known_dynamic_hit_references() -> None:
    extractor = CardFeatureExtractor.from_csv()

    barrage = extractor.extract("BARRAGE").hit_count_reference
    flechettes = extractor.extract("FLECHETTES").hit_count_reference
    tear_asunder = extractor.extract("TEAR_ASUNDER").hit_count_reference

    assert barrage is not None
    assert barrage.source is ScalingSource.COMBAT_COUNT
    assert barrage.filter_value == "ORB_COUNT"

    assert flechettes is not None
    assert flechettes.source is ScalingSource.UNPARSED
    assert "PileType.Hand" in flechettes.filter_value
    assert ".Cards.Count((CardMo" in flechettes.filter_value
    assert len(flechettes.filter_value) > 80

    assert tear_asunder is not None
    assert tear_asunder.source is ScalingSource.UNPARSED
    assert "RepeatVar(1)" in tear_asunder.filter_value
    assert "CalculatedHits" in tear_asunder.filter_value
    assert len(tear_asunder.filter_value) > 80


@pytest.mark.skipif(not DEFAULT_CARD_FEATURES_CSV.exists(), reason="card catalog not present")
def test_real_catalog_retains_regent_star_costs() -> None:
    extractor = CardFeatureExtractor.from_csv()

    astral_pulse = extractor.extract("ASTRAL_PULSE")
    falling_star = extractor.extract("FALLING_STAR")

    assert astral_pulse.cost == 0.0
    assert astral_pulse.star_cost == 3.0
    assert astral_pulse.uses_star_cost is True
    assert falling_star.cost == 0.0
    assert falling_star.star_cost == 2.0
    assert falling_star.uses_star_cost is True


@pytest.mark.skipif(not DEFAULT_CARD_FEATURES_CSV.exists(), reason="card catalog not present")
def test_real_catalog_loads() -> None:
    extractor = CardFeatureExtractor.from_csv()

    assert len(extractor) > 0
    for card_id in extractor.card_ids():
        assert len(extractor.extract(card_id).to_vector()) == len(FEATURE_NAMES)
