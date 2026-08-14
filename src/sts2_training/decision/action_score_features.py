"""Versioned pre-simulation features for learned Combat ``action_score`` ranking.

The feature contract combines the exact pre-action board vocabulary used by Combat
ValueModel with candidate-local public metadata. Raw decision-context main effects are
retained for inspection/schema lineage, while explicit context-by-candidate interactions
make shared board and choice context visible to the within-decision pairwise ranker.
Opaque action/card identifiers are never learned features; they are used only to resolve
public card/target metadata.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.decision.combat_observation import CombatObservation
from sts2_training.decision.value_features import VALUE_FEATURE_NAMES, combat_value_features
from sts2_training.selection.choice_semantics import choice_option_id, pending_choice_context

ACTION_SCORE_FEATURE_SCHEMA_VERSION = 4

_BOARD_FEATURE_NAMES = tuple(f"board_{name}" for name in VALUE_FEATURE_NAMES)
_CANDIDATE_FEATURE_NAMES = (
    "action_card",
    "action_potion",
    "action_system",
    "action_choice_target",
    "action_choice_card",
    "action_choice_confirm",
    "action_choice_skip",
    "action_other",
    "cost",
    "cost_present",
    "affordable",
    "energy_after",
    "target_self",
    "target_single_enemy",
    "target_all_enemies",
    "target_other",
    "card_present",
    "card_type_attack",
    "card_type_skill",
    "card_type_power",
    "card_type_curse",
    "card_type_status",
    "card_type_other",
    "card_rarity_basic",
    "card_rarity_common",
    "card_rarity_uncommon",
    "card_rarity_rare",
    "card_rarity_other",
    "card_upgrade_level",
    "card_upgraded",
    "card_enchanted",
    "card_enchantment_amount",
    "card_tinker_time",
    "potion_present",
    "potion_rarity_common",
    "potion_rarity_uncommon",
    "potion_rarity_rare",
    "potion_rarity_other",
    "target_hp_ratio",
    "target_block_ratio",
    "target_incoming_attack",
    "target_alive",
    "choice_selected_count",
    "choice_min_select",
    "choice_max_select",
    "choice_capacity_present",
    "choice_op_gain",
    "choice_op_discard",
    "choice_op_exhaust",
    "choice_op_upgrade",
    "choice_op_retrieve",
    "choice_op_play",
    "choice_op_replay",
    "choice_op_remove",
    "choice_op_transform",
    "choice_op_other",
)

# Raw board main effects cancel exactly in same-decision pairwise deltas. Interacting
# decision-level context with semantic candidate indicators lets the linear ranker learn
# preferences such as "favor Skill/Potion over Attack when danger is high".
_CONTEXT_BOARD_FEATURE_NAMES = (
    "board_player_hp_ratio",
    "board_player_block",
    "board_energy",
    "board_enemy_hp_ratio",
    "board_incoming_damage",
    "board_danger_ratio",
    "board_enemies_alive",
    "board_hand_size",
    "board_potion_count",
    "board_lethal_threat",
)
_CONTEXT_CANDIDATE_FEATURE_NAMES = (
    "action_card",
    "action_potion",
    "action_system",
    "action_choice_target",
    "action_choice_card",
    "action_choice_confirm",
    "action_choice_skip",
    "action_other",
    "card_type_attack",
    "card_type_skill",
    "card_type_power",
    "card_type_curse",
    "card_type_status",
    "card_type_other",
    "target_self",
    "target_single_enemy",
    "target_all_enemies",
    "target_other",
    "potion_present",
)
_BOARD_INTERACTION_FEATURE_NAMES = tuple(
    f"context_{board_name.removeprefix('board_')}_x_{candidate_name}"
    for board_name in _CONTEXT_BOARD_FEATURE_NAMES
    for candidate_name in _CONTEXT_CANDIDATE_FEATURE_NAMES
)

# Choice operation and selection state are also decision-level context, so their raw
# columns cancel in pairwise deltas. Cross them with option-local semantics so the ranker
# can learn operation-dependent preferences (gain/retrieve versus discard/exhaust, etc.).
_CHOICE_CONTEXT_FEATURE_NAMES = (
    "choice_selected_count",
    "choice_min_select",
    "choice_max_select",
    "choice_capacity_present",
    "choice_op_gain",
    "choice_op_discard",
    "choice_op_exhaust",
    "choice_op_upgrade",
    "choice_op_retrieve",
    "choice_op_play",
    "choice_op_replay",
    "choice_op_remove",
    "choice_op_transform",
    "choice_op_other",
)
_CHOICE_CONTEXT_CANDIDATE_FEATURE_NAMES = (
    "action_choice_card",
    "action_choice_confirm",
    "action_choice_skip",
    "card_type_attack",
    "card_type_skill",
    "card_type_power",
    "card_type_curse",
    "card_type_status",
    "card_type_other",
    "card_rarity_basic",
    "card_rarity_common",
    "card_rarity_uncommon",
    "card_rarity_rare",
    "card_rarity_other",
    "card_upgraded",
    "card_enchanted",
    "card_tinker_time",
)
_CHOICE_INTERACTION_FEATURE_NAMES = tuple(
    f"context_{choice_name}_x_{candidate_name}"
    for choice_name in _CHOICE_CONTEXT_FEATURE_NAMES
    for candidate_name in _CHOICE_CONTEXT_CANDIDATE_FEATURE_NAMES
)

ACTION_SCORE_FEATURE_NAMES: tuple[str, ...] = (
    _BOARD_FEATURE_NAMES
    + _CANDIDATE_FEATURE_NAMES
    + _BOARD_INTERACTION_FEATURE_NAMES
    + _CHOICE_INTERACTION_FEATURE_NAMES
)

_ACTION_TYPE_FEATURE = {
    "card": "action_card",
    "potion": "action_potion",
    "system": "action_system",
    "choice_target": "action_choice_target",
    "choice_card": "action_choice_card",
    "choice_confirm": "action_choice_confirm",
    "choice_skip": "action_choice_skip",
}
_CARD_TYPE_FEATURE = {
    "attack": "card_type_attack",
    "skill": "card_type_skill",
    "power": "card_type_power",
    "curse": "card_type_curse",
    "status": "card_type_status",
}
_CARD_RARITY_FEATURE = {
    "basic": "card_rarity_basic",
    "common": "card_rarity_common",
    "uncommon": "card_rarity_uncommon",
    "rare": "card_rarity_rare",
}
_POTION_RARITY_FEATURE = {
    "common": "potion_rarity_common",
    "uncommon": "potion_rarity_uncommon",
    "rare": "potion_rarity_rare",
}
_CHOICE_OPERATION_FEATURE = {
    operation: f"choice_op_{operation}"
    for operation in (
        "gain",
        "discard",
        "exhaust",
        "upgrade",
        "retrieve",
        "play",
        "replay",
        "remove",
        "transform",
    )
}


def combat_action_score_features(
    masked_emulator_dto: Mapping[str, Any],
    action: Mapping[str, Any],
) -> tuple[float, ...]:
    """Return the authoritative feature vector for one pre-simulation candidate action."""

    return combat_action_score_feature_matrix(masked_emulator_dto, (action,))[0]


def combat_action_score_feature_matrix(
    masked_emulator_dto: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
) -> list[tuple[float, ...]]:
    """Vectorize candidates while computing the shared board representation only once."""

    board = combat_value_features(masked_emulator_dto)
    observation = CombatObservation.from_dto(masked_emulator_dto, strict=True)
    board_values = {
        name: value for name, value in zip(_BOARD_FEATURE_NAMES, board, strict=True)
    }
    rows: list[tuple[float, ...]] = []
    for action in actions:
        values = {name: 0.0 for name in _CANDIDATE_FEATURE_NAMES}

        action_type = _string(action.get("action_type"))
        values[_ACTION_TYPE_FEATURE.get(action_type, "action_other")] = 1.0
        params = _mapping(action.get("parameters"))
        card = _candidate_card(action, masked_emulator_dto, observation)
        if card:
            _populate_card_features(values, card)

        cost = _finite_number(params.get("cost"))
        if cost is None:
            cost = _finite_number(card.get("cost"))
        if cost is not None:
            values["cost"] = cost
            values["cost_present"] = 1.0
            if observation.energy is not None:
                values["affordable"] = 1.0 if cost <= observation.energy else 0.0
                values["energy_after"] = observation.energy - cost

        target_type = _string(params.get("targetType")) or _string(card.get("targetType"))
        values[_target_feature(target_type)] = 1.0

        potion = _candidate_potion(params, observation)
        if potion:
            values["potion_present"] = 1.0
            rarity = (_string(potion.get("rarity")) or "").casefold()
            values[_POTION_RARITY_FEATURE.get(rarity, "potion_rarity_other")] = 1.0

        _populate_target_features(values, params, observation)
        _populate_choice_features(values, masked_emulator_dto)

        board_interaction_values = {
            f"context_{board_name.removeprefix('board_')}_x_{candidate_name}": (
                float(board_values[board_name]) * float(values[candidate_name])
            )
            for board_name in _CONTEXT_BOARD_FEATURE_NAMES
            for candidate_name in _CONTEXT_CANDIDATE_FEATURE_NAMES
        }
        choice_interaction_values = {
            f"context_{choice_name}_x_{candidate_name}": (
                float(values[choice_name]) * float(values[candidate_name])
            )
            for choice_name in _CHOICE_CONTEXT_FEATURE_NAMES
            for candidate_name in _CHOICE_CONTEXT_CANDIDATE_FEATURE_NAMES
        }
        feature_values = {
            **board_values,
            **values,
            **board_interaction_values,
            **choice_interaction_values,
        }
        rows.append(tuple(float(feature_values[name]) for name in ACTION_SCORE_FEATURE_NAMES))
    return rows


def _candidate_card(
    action: Mapping[str, Any],
    dto: Mapping[str, Any],
    observation: CombatObservation,
) -> Mapping[str, Any]:
    action_type = action.get("action_type")
    params = _mapping(action.get("parameters"))
    if action_type == "card":
        card_id = _string(params.get("cardId"))
        if card_id is None:
            return {}
        matches = [card for card in observation.hand if card.get("id") == card_id]
        if not matches:
            return {}
        if len(matches) == 1:
            return matches[0]
        action_cost = _finite_number(params.get("cost"))
        target_type = _string(params.get("targetType"))
        narrowed = [
            card
            for card in matches
            if (action_cost is None or _finite_number(card.get("cost")) == action_cost)
            and (target_type is None or _string(card.get("targetType")) == target_type)
        ]
        if len(narrowed) == 1:
            return narrowed[0]
        return _common_metadata(narrowed or matches)

    if action_type != "choice_card":
        return {}
    context = pending_choice_context(dto)
    if context is None or not context.identity_valid:
        return {}
    option_id = choice_option_id(action)
    if option_id is None:
        return {}
    pending = _mapping(dto.get("pendingChoice"))
    options = pending.get("options")
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes, bytearray)):
        return {}
    matches = [
        option
        for option in options
        if isinstance(option, Mapping) and option.get("optionId") == option_id
    ]
    return matches[0] if len(matches) == 1 else {}


def _populate_card_features(values: dict[str, float], card: Mapping[str, Any]) -> None:
    values["card_present"] = 1.0
    card_type = (_string(card.get("type")) or "").casefold()
    values[_CARD_TYPE_FEATURE.get(card_type, "card_type_other")] = 1.0
    rarity = (_string(card.get("rarity")) or "").casefold()
    values[_CARD_RARITY_FEATURE.get(rarity, "card_rarity_other")] = 1.0

    upgrade_level = _finite_number(card.get("upgradeLevel"))
    if upgrade_level is None:
        upgrade_level = 1.0 if card.get("upgraded") is True else 0.0
    upgrade_level = max(0.0, upgrade_level)
    values["card_upgrade_level"] = upgrade_level
    values["card_upgraded"] = 1.0 if upgrade_level > 0.0 else 0.0

    enchantment = card.get("enchantment")
    if isinstance(enchantment, Mapping):
        values["card_enchanted"] = 1.0
        amount = _finite_number(enchantment.get("amount"))
        if amount is not None:
            values["card_enchantment_amount"] = amount
    if card.get("tinkerTimeType") is not None or card.get("tinkerTimeRider") is not None:
        values["card_tinker_time"] = 1.0


def _candidate_potion(
    params: Mapping[str, Any], observation: CombatObservation
) -> Mapping[str, Any]:
    raw_slot = _finite_number(params.get("potionSlot"))
    if raw_slot is None or not raw_slot.is_integer():
        return {}
    slot = int(raw_slot)
    if slot < 0 or slot >= len(observation.potions):
        return {}
    return observation.potions[slot]


def _populate_target_features(
    values: dict[str, float],
    params: Mapping[str, Any],
    observation: CombatObservation,
) -> None:
    raw_index = _finite_number(params.get("enemyIndex"))
    enemy = None
    if raw_index is not None and raw_index.is_integer():
        enemy = observation.enemy_by_index(int(raw_index))

    hp = enemy.hp if enemy is not None else _finite_number(params.get("hp"))
    max_hp = enemy.max_hp if enemy is not None else _finite_number(params.get("maxHp"))
    block = enemy.block if enemy is not None else _finite_number(params.get("block"))
    if block is None:
        block = 0.0
    if hp is not None:
        denominator = max(1.0, hp if max_hp is None else max_hp)
        values["target_hp_ratio"] = max(0.0, hp) / denominator
        values["target_block_ratio"] = max(0.0, block) / denominator
        values["target_alive"] = 1.0 if (enemy.is_alive if enemy is not None else hp > 0.0) else 0.0
    if enemy is not None:
        values["target_incoming_attack"] = enemy.incoming_attack


def _populate_choice_features(
    values: dict[str, float], dto: Mapping[str, Any]
) -> None:
    pending = _mapping(dto.get("pendingChoice"))
    selected = _finite_number(pending.get("selectedCount"))
    min_select = _finite_number(pending.get("minSelect"))
    max_select = _finite_number(pending.get("maxSelect"))
    if selected is not None:
        values["choice_selected_count"] = selected
    if min_select is not None:
        values["choice_min_select"] = min_select
    if max_select is not None:
        values["choice_max_select"] = max_select
        values["choice_capacity_present"] = 1.0

    context = pending_choice_context(dto)
    operation = context.semantics.operation if context is not None and context.semantics.is_known else None
    values[_CHOICE_OPERATION_FEATURE.get(operation, "choice_op_other")] = 1.0


def _target_feature(target_type: str | None) -> str:
    normalized = "" if target_type is None else target_type.casefold().replace("_", "")
    if normalized in {"self", "player"}:
        return "target_self"
    if normalized in {"singleenemy", "anyenemy", "enemy"}:
        return "target_single_enemy"
    if normalized in {"allenemies", "allenemy"}:
        return "target_all_enemies"
    return "target_other"


def _common_metadata(cards: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not cards:
        return {}
    common: dict[str, Any] = {}
    keys = set(cards[0])
    for card in cards[1:]:
        keys &= set(card)
    for key in keys:
        first = cards[0].get(key)
        if all(card.get(key) == first for card in cards[1:]):
            common[key] = first
    return common


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


__all__ = [
    "ACTION_SCORE_FEATURE_NAMES",
    "ACTION_SCORE_FEATURE_SCHEMA_VERSION",
    "combat_action_score_feature_matrix",
    "combat_action_score_features",
]
