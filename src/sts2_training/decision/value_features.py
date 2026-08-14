"""Versioned features for learned Combat ValueModel inference.

Feature schema v2 consumes STS2_RL mask v1.2 card identity. Pile order remains hidden,
but upgrade level, enchantment, tinker-time state, semantic card type, and multiset count
are public information and must affect board evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sts2_training.decision.combat_observation import CombatObservation
from sts2_training.decision.oracle_log import (
    ORACLE_VALUE_MASK_VERSION,
    require_oracle_value_mask_version,
)

VALUE_FEATURE_SCHEMA_VERSION = 2
VALUE_FEATURE_NAMES: tuple[str, ...] = (
    "player_hp_ratio",
    "player_block",
    "energy",
    "energy_present",
    "enemy_hp_ratio",
    "incoming_damage",
    "danger_ratio",
    "enemies_alive",
    "hand_size",
    "potion_count",
    "player_power_stacks",
    "enemy_power_stacks",
    "lethal_threat",
    "draw_pile_size",
    "discard_pile_size",
    "exhaust_pile_size",
    "known_card_count",
    "upgraded_card_count",
    "upgrade_level_sum",
    "max_upgrade_level",
    "enchanted_card_count",
    "enchantment_amount_sum",
    "tinker_time_card_count",
    "attack_card_count",
    "skill_card_count",
    "power_card_count",
    "upgraded_attack_count",
    "upgraded_skill_count",
    "enchanted_attack_count",
    "enchanted_skill_count",
    "hand_upgraded_card_count",
    "hand_upgrade_level_sum",
    "hand_enchanted_card_count",
    "hand_enchantment_amount_sum",
)


@dataclass(frozen=True)
class CombatCardSummary:
    hand_size: int = 0
    draw_pile_size: int = 0
    discard_pile_size: int = 0
    exhaust_pile_size: int = 0
    known_card_count: int = 0
    upgraded_card_count: int = 0
    upgrade_level_sum: int = 0
    max_upgrade_level: int = 0
    enchanted_card_count: int = 0
    enchantment_amount_sum: int = 0
    tinker_time_card_count: int = 0
    attack_card_count: int = 0
    skill_card_count: int = 0
    power_card_count: int = 0
    upgraded_attack_count: int = 0
    upgraded_skill_count: int = 0
    enchanted_attack_count: int = 0
    enchanted_skill_count: int = 0
    hand_upgraded_card_count: int = 0
    hand_upgrade_level_sum: int = 0
    hand_enchanted_card_count: int = 0
    hand_enchantment_amount_sum: int = 0


@dataclass
class _MutableCardSummary:
    hand_size: int = 0
    draw_pile_size: int = 0
    discard_pile_size: int = 0
    exhaust_pile_size: int = 0
    known_card_count: int = 0
    upgraded_card_count: int = 0
    upgrade_level_sum: int = 0
    max_upgrade_level: int = 0
    enchanted_card_count: int = 0
    enchantment_amount_sum: int = 0
    tinker_time_card_count: int = 0
    attack_card_count: int = 0
    skill_card_count: int = 0
    power_card_count: int = 0
    upgraded_attack_count: int = 0
    upgraded_skill_count: int = 0
    enchanted_attack_count: int = 0
    enchanted_skill_count: int = 0
    hand_upgraded_card_count: int = 0
    hand_upgrade_level_sum: int = 0
    hand_enchanted_card_count: int = 0
    hand_enchantment_amount_sum: int = 0


def combat_card_summary(dto: Mapping[str, Any]) -> CombatCardSummary:
    """Summarize full mask-v1.2 card identity without using opaque card ids as features."""

    require_oracle_value_mask_version(dto, context="Combat Value features")
    summary = _MutableCardSummary()
    for zone, card, count in _iter_cards(dto):
        upgrade_level = _required_nonnegative_int(
            card.get("upgradeLevel"), f"{zone}.upgradeLevel"
        )
        enchantment = _enchantment(card.get("enchantment"), f"{zone}.enchantment")
        card_type = card.get("type")
        if card_type is not None and not isinstance(card_type, str):
            raise ValueError(f"{zone}.type must be a string when provided")
        normalized_type = "" if card_type is None else card_type.casefold()
        upgraded = upgrade_level > 0
        enchanted = enchantment is not None
        enchantment_amount = 0 if enchantment is None else enchantment[1]
        tinker = card.get("tinkerTimeType") is not None or card.get("tinkerTimeRider") is not None

        summary.known_card_count += count
        summary.upgrade_level_sum += upgrade_level * count
        summary.max_upgrade_level = max(summary.max_upgrade_level, upgrade_level)
        if upgraded:
            summary.upgraded_card_count += count
        if enchanted:
            summary.enchanted_card_count += count
            summary.enchantment_amount_sum += enchantment_amount * count
        if tinker:
            summary.tinker_time_card_count += count

        if normalized_type == "attack":
            summary.attack_card_count += count
            if upgraded:
                summary.upgraded_attack_count += count
            if enchanted:
                summary.enchanted_attack_count += count
        elif normalized_type == "skill":
            summary.skill_card_count += count
            if upgraded:
                summary.upgraded_skill_count += count
            if enchanted:
                summary.enchanted_skill_count += count
        elif normalized_type == "power":
            summary.power_card_count += count

        if zone == "hand":
            summary.hand_size += count
            summary.hand_upgrade_level_sum += upgrade_level * count
            if upgraded:
                summary.hand_upgraded_card_count += count
            if enchanted:
                summary.hand_enchanted_card_count += count
                summary.hand_enchantment_amount_sum += enchantment_amount * count
        elif zone == "drawPile":
            summary.draw_pile_size += count
        elif zone == "discardPile":
            summary.discard_pile_size += count
        elif zone == "exhaustPile":
            summary.exhaust_pile_size += count

    return CombatCardSummary(**summary.__dict__)


def combat_value_features(dto: Mapping[str, Any]) -> tuple[float, ...]:
    """Return the authoritative v2 feature vector for one mask-v1.2 ValueModel DTO."""

    cards = combat_card_summary(dto)
    observation = CombatObservation.from_dto(dto, strict=True)
    enemies = observation.enemies
    alive = observation.alive_enemies
    enemy_hp = sum(enemy.hp for enemy in enemies)
    enemy_max_hp = sum(enemy.max_hp for enemy in enemies) or 1.0
    energy = 0.0 if observation.energy is None else observation.energy

    values = {
        "player_hp_ratio": observation.hp / observation.max_hp,
        "player_block": observation.block,
        "energy": energy,
        "energy_present": 0.0 if observation.energy is None else 1.0,
        "enemy_hp_ratio": enemy_hp / enemy_max_hp,
        "incoming_damage": observation.incoming_damage,
        "danger_ratio": observation.danger_ratio,
        "enemies_alive": float(len(alive)),
        "hand_size": float(cards.hand_size),
        "potion_count": float(len(observation.potions)),
        "player_power_stacks": _power_stack_amount(observation.player_powers),
        "enemy_power_stacks": sum(_power_stack_amount(enemy.powers) for enemy in alive),
        "lethal_threat": 1.0 if observation.lethal_threat else 0.0,
        **{
            name: float(getattr(cards, name))
            for name in VALUE_FEATURE_NAMES
            if hasattr(cards, name)
        },
    }
    return tuple(float(values[name]) for name in VALUE_FEATURE_NAMES)


def _iter_cards(dto: Mapping[str, Any]):
    hand = _mapping_sequence(dto.get("hand"), "hand")
    for index, card in enumerate(hand):
        _require_card_id(card, f"hand[{index}]")
        yield "hand", card, 1

    for pile_name in ("drawPile", "discardPile", "exhaustPile"):
        pile = _mapping_sequence(dto.get(pile_name), pile_name)
        for index, card in enumerate(pile):
            field = f"{pile_name}[{index}]"
            _require_card_id(card, field)
            count = _required_positive_int(card.get("count"), f"{field}.count")
            yield pile_name, card, count


def _mapping_sequence(value: Any, field: str) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"Combat Value input {field} must be a sequence of mappings")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"Combat Value input {field}[{index}] must be a mapping")
        result.append(item)
    return result


def _require_card_id(card: Mapping[str, Any], field: str) -> None:
    value = card.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Combat Value input {field}.id must be a non-empty string")


def _required_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Combat Value input {field} must be a non-negative integer")
    return value


def _required_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Combat Value input {field} must be a positive integer")
    return value


def _enchantment(value: Any, field: str) -> tuple[str, int, str | None] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"Combat Value input {field} must be a mapping or null")
    enchantment_id = value.get("id")
    amount = value.get("amount")
    status = value.get("status")
    if not isinstance(enchantment_id, str) or not enchantment_id:
        raise ValueError(f"Combat Value input {field}.id must be a non-empty string")
    if isinstance(amount, bool) or not isinstance(amount, int):
        raise ValueError(f"Combat Value input {field}.amount must be an integer")
    if status is not None and not isinstance(status, str):
        raise ValueError(f"Combat Value input {field}.status must be a string when provided")
    return enchantment_id, amount, status


def _power_stack_amount(powers: Sequence[Mapping[str, Any]]) -> float:
    total = 0.0
    for power in powers:
        amount = power.get("amount", 1.0)
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            continue
        total += abs(float(amount))
    return total


__all__ = [
    "CombatCardSummary",
    "ORACLE_VALUE_MASK_VERSION",
    "VALUE_FEATURE_NAMES",
    "VALUE_FEATURE_SCHEMA_VERSION",
    "combat_card_summary",
    "combat_value_features",
]
