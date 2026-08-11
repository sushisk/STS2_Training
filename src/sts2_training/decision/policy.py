"""Policy candidate generation for combat beam search.

`PolicyModel` defines the learned-policy seam. `PriorHeuristicPolicy` is the
model-free bootstrap implementation: it uses only information already exposed in
`masked_emulator_dto`/`legal_actions` to rank strategically useful branches before
expensive emulator/value-function evaluation.

The heuristic is intentionally coarse. It does not try to reproduce card rules in
Python; the Emulator remains authoritative. Its job is branch recall: keep a useful
mix of offense, defense/utility, potions, target choices, and End Turn inside a small
`top_k`, so beam search spends its simulations on stronger candidates.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sts2_training.selection.action_classification import (
    CARD_ACTION_TYPE,
    CHOICE_CARD_ACTION_TYPE,
    CHOICE_CONFIRM_ACTION_TYPE,
    CHOICE_SKIP_ACTION_TYPE,
)

JsonObject = Mapping[str, Any]

_SYSTEM_ACTION_TYPE = "system"
_POTION_ACTION_TYPE = "potion"
_CHOICE_TARGET_ACTION_TYPE = "choice_target"

_ACTION_TYPE_BASE_SCORE: dict[str, float] = {
    _CHOICE_TARGET_ACTION_TYPE: 80.0,
    CARD_ACTION_TYPE: 60.0,
    CHOICE_CARD_ACTION_TYPE: 55.0,
    CHOICE_CONFIRM_ACTION_TYPE: 32.0,
    CHOICE_SKIP_ACTION_TYPE: 12.0,
    _POTION_ACTION_TYPE: 30.0,
    _SYSTEM_ACTION_TYPE: 10.0,
}

_CARD_TYPE_SCORE = {
    "Attack": 8.0,
    "Skill": 7.0,
    "Power": 6.0,
    "Curse": -50.0,
    "Status": -35.0,
}

_RARITY_SCORE = {
    "Rare": 3.0,
    "Uncommon": 1.5,
    "Common": 0.5,
}

_CHOICE_CARD_TYPE_SCORE = {
    "Attack": 1.5,
    "Skill": 1.0,
    "Power": 0.5,
    "Curse": -100.0,
    "Status": -50.0,
}

_CHOICE_RARITY_SCORE = {
    "Rare": 4.0,
    "Uncommon": 2.0,
    "Common": 1.0,
}


@dataclass(frozen=True)
class ActionCandidate:
    """One action a `PolicyModel` proposes for beam search to branch on."""

    action_id: str


@dataclass(frozen=True)
class _CombatContext:
    energy: float | None
    incoming_damage: float
    danger_ratio: float
    lethal_threat: bool
    enemies_alive: int
    playable_cards: int


class PolicyModel:
    """Proposes candidate actions for one decision (`legal_actions` plus the
    full `masked_emulator_dto` they came from), best-first, capped at `top_k`.

    Implement `propose` for scalar inference, or override `propose_batch`
    directly for a batch-only learned model. A batch-only model does not need
    to provide a dummy scalar implementation merely to satisfy an abstract
    base-class contract.
    """

    def propose(
        self,
        legal_actions: Sequence[JsonObject],
        masked_emulator_dto: Mapping[str, Any],
        *,
        top_k: int,
    ) -> list[ActionCandidate]:
        raise NotImplementedError("PolicyModel must override propose or propose_batch")

    def propose_batch(
        self,
        requests: Sequence[tuple[Sequence[JsonObject], Mapping[str, Any]]],
        *,
        top_k: int,
    ) -> list[list[ActionCandidate]]:
        """Batched counterpart of `propose`, one entry per request, in order.

        Override this directly in a learned policy for real batched inference;
        the default here is a plain loop and carries none of the throughput
        benefit the batched call is meant to provide.
        """
        return [
            self.propose(legal_actions, dto, top_k=top_k) for legal_actions, dto in requests
        ]


class PriorHeuristicPolicy(PolicyModel):
    """Cheap, state-aware branch prior used before a learned policy exists.

    The scorer deliberately consumes only public DTO fields. It favors cards that fit
    the current danger level, promotes potions when incoming damage is severe, ranks
    target choices by killability and enemy attack intent, and understands basic
    pending-card-choice quality. It never reimplements card effects; beam simulation
    and the value function remain authoritative.

    For ordinary combat decisions, one End Turn/system action is retained whenever
    `top_k >= 2`. This prevents a hand with many playable cards from crowding the
    turn-boundary branch out of the beam entirely. `rng`, when supplied, randomizes
    only equal-score ties rather than discarding the heuristic ordering.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng

    def propose(
        self,
        legal_actions: Sequence[JsonObject],
        masked_emulator_dto: Mapping[str, Any],
        *,
        top_k: int,
    ) -> list[ActionCandidate]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        # Filter in exactly one pass. Besides avoiding redundant work, this keeps the
        # availability contract simple for proxy/mocked Mapping implementations.
        available: list[JsonObject] = []
        for action in legal_actions:
            if action.get("is_available") is False:
                continue
            if not isinstance(action.get("action_type"), str):
                continue
            available.append(action)
        if not available:
            return []

        context = _combat_context(masked_emulator_dto, available)
        choice_card_positions: dict[int, int] = {}
        choice_position = 0
        for action in available:
            if action.get("action_type") == CHOICE_CARD_ACTION_TYPE:
                choice_card_positions[id(action)] = choice_position
                choice_position += 1

        scored: list[tuple[float, float, int, JsonObject]] = []
        for index, action in enumerate(available):
            score = _score_action(
                action,
                masked_emulator_dto,
                context,
                choice_card_position=choice_card_positions.get(id(action)),
            )
            tie_break = self._rng.random() if self._rng is not None else 0.0
            scored.append((score, tie_break, index, action))

        if self._rng is None:
            scored.sort(key=lambda row: (-row[0], row[2]))
        else:
            scored.sort(key=lambda row: (-row[0], -row[1], row[2]))

        selected = scored[:top_k]

        # End Turn is strategically different from "play another card" and is the
        # transition that exposes enemy-turn consequences. Reserve one slot for it in
        # a real combat-action decision, matching the turn-beam design requirement.
        if top_k >= 2 and _is_regular_combat_action_set(available):
            best_system = next(
                (row for row in scored if row[3].get("action_type") == _SYSTEM_ACTION_TYPE),
                None,
            )
            if best_system is not None and best_system not in selected:
                selected = selected[:-1] + [best_system]
                selected.sort(key=lambda row: scored.index(row))

        return [ActionCandidate(action_id=action["action_id"]) for _, _, _, action in selected]


def _score_action(
    action: JsonObject,
    dto: Mapping[str, Any],
    context: _CombatContext,
    *,
    choice_card_position: int | None,
) -> float:
    action_type = action.get("action_type")
    score = _ACTION_TYPE_BASE_SCORE.get(action_type, -10.0)

    if action_type == CARD_ACTION_TYPE:
        return score + _score_playable_card(action, dto, context)
    if action_type == _POTION_ACTION_TYPE:
        return score + _score_potion(action, dto, context)
    if action_type == _SYSTEM_ACTION_TYPE:
        return score + _score_end_turn(context)
    if action_type == _CHOICE_TARGET_ACTION_TYPE:
        return score + _score_target(action, dto)
    if action_type == CHOICE_CARD_ACTION_TYPE:
        return score + _score_choice_card(action, dto, choice_card_position)
    if action_type in (CHOICE_CONFIRM_ACTION_TYPE, CHOICE_SKIP_ACTION_TYPE):
        return score + _score_choice_completion(action_type, dto)
    return score


def _score_playable_card(
    action: JsonObject,
    dto: Mapping[str, Any],
    context: _CombatContext,
) -> float:
    params = _mapping(action.get("parameters"))
    card = _hand_card_for(action, dto)
    card_type = _string(card.get("type"))
    rarity = _string(card.get("rarity"))
    card_id = _string(params.get("cardId")) or _string(card.get("id"))
    target_type = _string(params.get("targetType")) or _string(card.get("targetType"))

    score = _CARD_TYPE_SCORE.get(card_type, 0.0) + _RARITY_SCORE.get(rarity, 0.0)
    if card.get("upgraded") is True:
        score += 2.0
    upgrade_level = _finite_number(card.get("upgradeLevel"))
    if upgrade_level is not None and upgrade_level > 1:
        score += min(2.0, 0.5 * (upgrade_level - 1.0))

    cost = _finite_number(params.get("cost"))
    if cost is None:
        cost = _finite_number(card.get("cost"))
    if cost is not None:
        # Cheap actions are easier to combine in one turn, but keep this bonus small so
        # expensive high-impact cards can still win on card-type/threat context.
        score += max(0.0, 2.0 - cost) * 1.5
        if context.energy is not None and cost > context.energy:
            score -= 100.0  # defensive guard; normally such a card is not available.

    danger = min(1.0, context.danger_ratio)
    if card_type == "Skill":
        score += 12.0 * danger
    elif card_type == "Power":
        score += 7.0 * (1.0 - danger) - 10.0 * danger
    elif card_type == "Attack":
        score += 4.0 * (1.0 - danger)

    # Starter Defends are known defensive cards across characters. This narrow hint is
    # intentionally much smaller than an effect table: the Emulator still decides what
    # the card actually does; it only helps the prior react to obvious lethal pressure.
    if card_id is not None and card_id.startswith("DEFEND"):
        score += 10.0 * danger

    if target_type == "AllEnemies" and context.enemies_alive > 1:
        score += 3.0 * min(3, context.enemies_alive - 1)

    return score


def _score_potion(
    action: JsonObject,
    dto: Mapping[str, Any],
    context: _CombatContext,
) -> float:
    # Potions are consumable, so they sit below normal cards in safe states. As damage
    # pressure rises, ensure beam search actually explores spending one rather than
    # losing while preserving it.
    danger = min(1.0, context.danger_ratio)
    score = 30.0 * danger
    if context.lethal_threat:
        score += 15.0

    params = _mapping(action.get("parameters"))
    target_type = _string(params.get("targetType"))
    if target_type == "AllEnemies" and context.enemies_alive > 1:
        score += 3.0 * min(3, context.enemies_alive - 1)

    potion = _potion_for(action, dto)
    rarity = _string(potion.get("rarity"))
    # Small conservation prior only; emergency pressure above dominates it.
    if rarity == "Rare":
        score -= 4.0
    elif rarity == "Uncommon":
        score -= 2.0
    return score


def _score_end_turn(context: _CombatContext) -> float:
    score = 0.0
    if context.playable_cards == 0:
        score += 30.0
    if context.energy is not None:
        if context.energy <= 0:
            score += 20.0
        elif context.playable_cards > 0:
            score -= min(12.0, 4.0 * context.energy)
    if context.incoming_damage <= 0:
        score += 5.0
    if context.lethal_threat:
        score -= 35.0
    return score


def _score_target(action: JsonObject, dto: Mapping[str, Any]) -> float:
    params = _mapping(action.get("parameters"))
    hp = max(0.0, _finite_number(params.get("hp")) or 0.0)
    block = max(0.0, _finite_number(params.get("block")) or 0.0)
    max_hp = max(1.0, _finite_number(params.get("maxHp")) or max(1.0, hp))
    effective_hp_ratio = min(2.0, (hp + block) / max_hp)

    enemy = _enemy_for_target(params, dto)
    incoming = _enemy_attack(enemy)

    # Prefer targets that are closer to removal, with a substantial bonus for removing
    # enemies currently representing immediate incoming damage.
    return 20.0 * (1.0 - min(1.0, effective_hp_ratio)) + min(25.0, incoming * 0.75)


def _score_choice_card(
    action: JsonObject,
    dto: Mapping[str, Any],
    choice_card_position: int | None,
) -> float:
    card = _choice_option_for(action, dto, choice_card_position)
    if not card:
        return 0.0

    score = _CHOICE_RARITY_SCORE.get(_string(card.get("rarity")), 0.0)
    score += _CHOICE_CARD_TYPE_SCORE.get(_string(card.get("type")), 0.0)
    if card.get("upgraded") is True:
        score += 2.0
    cost = _finite_number(card.get("cost"))
    if cost is not None:
        score += max(0.0, 3.0 - cost)
    return score


def _score_choice_completion(action_type: str, dto: Mapping[str, Any]) -> float:
    pending = _mapping(dto.get("pendingChoice"))
    selected = _finite_number(pending.get("selectedCount")) or 0.0
    min_select = _finite_number(pending.get("minSelect")) or 0.0
    max_select = _finite_number(pending.get("maxSelect"))

    if selected < min_select:
        return -100.0
    if max_select is not None and selected >= max_select:
        return 40.0 if action_type == CHOICE_CONFIRM_ACTION_TYPE else 20.0
    if action_type == CHOICE_CONFIRM_ACTION_TYPE:
        return 8.0
    return 0.0


def _combat_context(
    dto: Mapping[str, Any], available: Sequence[JsonObject]
) -> _CombatContext:
    hp = max(0.0, _finite_number(dto.get("hp")) or 0.0)
    block = max(0.0, _finite_number(dto.get("block")) or 0.0)
    energy = _finite_number(dto.get("energy"))

    enemies = [
        enemy
        for enemy in _mapping_sequence(dto.get("enemies"))
        if enemy.get("isAlive") is not False
    ]
    incoming_before_block = sum(_enemy_attack(enemy) for enemy in enemies)
    incoming_damage = max(0.0, incoming_before_block - block)
    danger_ratio = incoming_damage / max(1.0, hp)

    playable_cards = sum(1 for action in available if action.get("action_type") == CARD_ACTION_TYPE)
    return _CombatContext(
        energy=energy,
        incoming_damage=incoming_damage,
        danger_ratio=danger_ratio,
        lethal_threat=hp > 0 and incoming_damage >= hp,
        enemies_alive=len(enemies),
        playable_cards=playable_cards,
    )


def _hand_card_for(action: JsonObject, dto: Mapping[str, Any]) -> Mapping[str, Any]:
    # `action_id` is deliberately opaque on the Training wire. Match against the
    # public semantic parameters instead of relying on the Emulator's current numeric
    # hand-index implementation detail. Duplicate copies normally share type/rarity;
    # cost/target matching narrows the rare per-instance differences.
    params = _mapping(action.get("parameters"))
    card_id = _string(params.get("cardId"))
    if card_id is None:
        return {}

    matches = [
        card for card in _mapping_sequence(dto.get("hand")) if card.get("id") == card_id
    ]
    if not matches:
        return {}
    if len(matches) == 1:
        return matches[0]

    action_cost = _finite_number(params.get("cost"))
    target_type = _string(params.get("targetType"))
    for card in matches:
        card_cost = _finite_number(card.get("cost"))
        card_target = _string(card.get("targetType"))
        if (action_cost is None or card_cost == action_cost) and (
            target_type is None or card_target == target_type
        ):
            return card
    return matches[0]


def _potion_for(action: JsonObject, dto: Mapping[str, Any]) -> Mapping[str, Any]:
    params = _mapping(action.get("parameters"))
    slot_number = _finite_number(params.get("potionSlot"))
    potions = dto.get("potions")
    if (
        slot_number is None
        or not isinstance(potions, Sequence)
        or isinstance(potions, (str, bytes))
    ):
        return {}
    slot = int(slot_number)
    if slot < 0 or slot >= len(potions):
        return {}
    potion = potions[slot]
    return potion if isinstance(potion, Mapping) else {}


def _enemy_for_target(params: Mapping[str, Any], dto: Mapping[str, Any]) -> Mapping[str, Any]:
    target_index = _finite_number(params.get("enemyIndex"))
    if target_index is None:
        return {}
    for enemy in _mapping_sequence(dto.get("enemies")):
        enemy_index = _finite_number(enemy.get("index"))
        if enemy_index is not None and int(enemy_index) == int(target_index):
            return enemy
    return {}


def _choice_option_for(
    action: JsonObject,
    dto: Mapping[str, Any],
    choice_card_position: int | None,
) -> Mapping[str, Any]:
    options = _mapping_sequence(_mapping(dto.get("pendingChoice")).get("options"))
    if choice_card_position is not None and 0 <= choice_card_position < len(options):
        return options[choice_card_position]

    params = _mapping(action.get("parameters"))
    card_id = _string(params.get("cardId")) or _string(action.get("label"))
    if card_id is not None:
        for option in options:
            if option.get("id") == card_id:
                return option
    return {}


def _enemy_attack(enemy: Mapping[str, Any]) -> float:
    intent = _mapping(enemy.get("intent"))
    damage = max(0.0, _finite_number(intent.get("attackDamage")) or 0.0)
    repeats_value = _finite_number(intent.get("attackRepeats"))
    repeats = max(0.0, 1.0 if repeats_value is None else repeats_value)
    return damage * repeats


def _is_regular_combat_action_set(actions: Sequence[JsonObject]) -> bool:
    action_types = {action.get("action_type") for action in actions}
    return _SYSTEM_ACTION_TYPE in action_types and bool(
        action_types & {CARD_ACTION_TYPE, _POTION_ACTION_TYPE}
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_sequence(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None
