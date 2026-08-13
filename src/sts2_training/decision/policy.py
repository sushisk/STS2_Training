"""Policy candidate generation for combat beam search.

`PolicyModel` defines the learned-policy seam. `PriorHeuristicPolicy` is the model-free
bootstrap implementation. Wire/schema normalization is shared with Value through
`CombatObservation`; policy-specific code only applies pre-simulation ``action_score``
ranking judgments. Structural branch-retention constraints are applied separately by the
Combat engine's candidate-coverage layer so replacing this policy with a learned model
does not change Beam topology.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sts2_training.decision.combat_observation import CombatObservation
from sts2_training.selection.action_classification import (
    CARD_ACTION_TYPE,
    CHOICE_CARD_ACTION_TYPE,
    CHOICE_CONFIRM_ACTION_TYPE,
    CHOICE_SKIP_ACTION_TYPE,
)
from sts2_training.selection.choice_card_heuristic import choice_card_preference_scores

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


@dataclass(frozen=True)
class ActionCandidate:
    """One action a `PolicyModel` proposes, with its optional ``action_score``."""

    action_id: str
    action_score: float | None = None

    @property
    def action_rank(self) -> int | None:
        """Policy candidates acquire a concrete rank only after proposal ordering."""

        return None


@dataclass(frozen=True)
class _CombatContext:
    observation: CombatObservation
    playable_cards: int
    choice_card_scores: Mapping[str, float]


class PolicyModel:
    """Ranks candidate actions for one decision, best-first, capped at `top_k`.

    Structural coverage is deliberately not part of this contract. The candidate layer
    may add required branch types after ranking so learned policies remain drop-in
    replacements for the heuristic prior.

    Implementations may populate ``ActionCandidate.action_score`` with the cheap scalar
    used for this pre-simulation ranking. The field is optional so policies that expose
    only an ordering remain valid.

    ``oracle_provenance`` is the optional configuration-lineage seam used by budgeted
    teacher collection. Implementations should return JSON-serializable metadata that
    distinguishes materially different model/checkpoint/configuration states.
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
        return [
            self.propose(legal_actions, dto, top_k=top_k) for legal_actions, dto in requests
        ]

    def oracle_provenance(self) -> Mapping[str, Any]:
        return {}


class PriorHeuristicPolicy(PolicyModel):
    """Cheap, state-aware action prior used before a learned policy exists.

    The scorer consumes the shared normalized `CombatObservation`. It favors cards that
    fit current danger, promotes potions under pressure, ranks targets using killability
    and normalized enemy attack intent, and uses only canonical v1 `choice_card`
    semantics/`optionId` identity for card-choice quality. Unknown, malformed, future, or
    inconsistent choice metadata remains neutral. `rng`, when supplied, randomizes only
    equal-action-score ties.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng

    def oracle_provenance(self) -> Mapping[str, Any]:
        if self._rng is None:
            return {"tie_break_rng": "disabled"}
        state_digest = hashlib.sha256(
            repr(self._rng.getstate()).encode("utf-8")
        ).hexdigest()
        return {
            "tie_break_rng": "python_random",
            "rng_state_sha256": state_digest,
        }

    def propose(
        self,
        legal_actions: Sequence[JsonObject],
        masked_emulator_dto: Mapping[str, Any],
        *,
        top_k: int,
    ) -> list[ActionCandidate]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        available: list[JsonObject] = []
        for action in legal_actions:
            if action.get("is_available") is False:
                continue
            if not isinstance(action.get("action_type"), str):
                continue
            available.append(action)
        if not available:
            return []

        observation = CombatObservation.from_dto(masked_emulator_dto)
        context = _CombatContext(
            observation=observation,
            playable_cards=sum(
                1 for action in available if action.get("action_type") == CARD_ACTION_TYPE
            ),
            choice_card_scores=choice_card_preference_scores(
                available, masked_emulator_dto
            ),
        )
        scored_actions: list[tuple[float, float, int, JsonObject]] = []
        for index, action in enumerate(available):
            action_score = _action_score(action, masked_emulator_dto, context)
            tie_break = self._rng.random() if self._rng is not None else 0.0
            scored_actions.append((action_score, tie_break, index, action))

        if self._rng is None:
            scored_actions.sort(key=lambda row: (-row[0], row[2]))
        else:
            scored_actions.sort(key=lambda row: (-row[0], -row[1], row[2]))

        return [
            ActionCandidate(action_id=action["action_id"], action_score=action_score)
            for action_score, _, _, action in scored_actions[:top_k]
        ]


def _action_score(
    action: JsonObject,
    dto: Mapping[str, Any],
    context: _CombatContext,
) -> float:
    action_type = action.get("action_type")
    action_score = _ACTION_TYPE_BASE_SCORE.get(action_type, -10.0)

    if action_type == CARD_ACTION_TYPE:
        return action_score + _score_playable_card(action, context)
    if action_type == _POTION_ACTION_TYPE:
        return action_score + _score_potion(action, context)
    if action_type == _SYSTEM_ACTION_TYPE:
        return action_score + _score_end_turn(context)
    if action_type == _CHOICE_TARGET_ACTION_TYPE:
        return action_score + _score_target(action, context.observation)
    if action_type == CHOICE_CARD_ACTION_TYPE:
        action_id = action.get("action_id")
        if isinstance(action_id, str):
            return action_score + context.choice_card_scores.get(action_id, 0.0)
        return action_score
    if action_type in (CHOICE_CONFIRM_ACTION_TYPE, CHOICE_SKIP_ACTION_TYPE):
        return action_score + _score_choice_completion(action_type, dto)
    return action_score


def _score_playable_card(action: JsonObject, context: _CombatContext) -> float:
    observation = context.observation
    params = _mapping(action.get("parameters"))
    card = _hand_card_for(action, observation)
    card_type = _string(card.get("type"))
    rarity = _string(card.get("rarity"))
    card_id = _string(params.get("cardId")) or _string(card.get("id"))
    target_type = _string(params.get("targetType")) or _string(card.get("targetType"))

    action_score = _CARD_TYPE_SCORE.get(card_type, 0.0) + _RARITY_SCORE.get(rarity, 0.0)
    if card.get("upgraded") is True:
        action_score += 2.0
    upgrade_level = _finite_number(card.get("upgradeLevel"))
    if upgrade_level is not None and upgrade_level > 1:
        action_score += min(2.0, 0.5 * (upgrade_level - 1.0))

    cost = _finite_number(params.get("cost"))
    if cost is None:
        cost = _finite_number(card.get("cost"))
    if cost is not None:
        action_score += max(0.0, 2.0 - cost) * 1.5
        if observation.energy is not None and cost > observation.energy:
            action_score -= 100.0

    danger = min(1.0, observation.danger_ratio)
    if card_type == "Skill":
        action_score += 12.0 * danger
    elif card_type == "Power":
        action_score += 7.0 * (1.0 - danger) - 10.0 * danger
    elif card_type == "Attack":
        action_score += 4.0 * (1.0 - danger)

    if card_id is not None and card_id.startswith("DEFEND"):
        action_score += 10.0 * danger

    enemies_alive = len(observation.alive_enemies)
    if target_type == "AllEnemies" and enemies_alive > 1:
        action_score += 3.0 * min(3, enemies_alive - 1)

    return action_score


def _score_potion(action: JsonObject, context: _CombatContext) -> float:
    observation = context.observation
    danger = min(1.0, observation.danger_ratio)
    action_score = 30.0 * danger
    if observation.lethal_threat:
        action_score += 15.0

    params = _mapping(action.get("parameters"))
    target_type = _string(params.get("targetType"))
    enemies_alive = len(observation.alive_enemies)
    if target_type == "AllEnemies" and enemies_alive > 1:
        action_score += 3.0 * min(3, enemies_alive - 1)

    potion = _potion_for(action, observation)
    rarity = _string(potion.get("rarity"))
    if rarity == "Rare":
        action_score -= 4.0
    elif rarity == "Uncommon":
        action_score -= 2.0
    return action_score


def _score_end_turn(context: _CombatContext) -> float:
    observation = context.observation
    action_score = 0.0
    if context.playable_cards == 0:
        action_score += 30.0
    if observation.energy is not None:
        if observation.energy <= 0:
            action_score += 20.0
        elif context.playable_cards > 0:
            action_score -= min(12.0, 4.0 * observation.energy)
    if observation.incoming_damage <= 0:
        action_score += 5.0
    if observation.lethal_threat:
        action_score -= 35.0
    return action_score


def _score_target(action: JsonObject, observation: CombatObservation) -> float:
    params = _mapping(action.get("parameters"))
    hp = max(0.0, _finite_number(params.get("hp")) or 0.0)
    block = max(0.0, _finite_number(params.get("block")) or 0.0)
    max_hp = max(1.0, _finite_number(params.get("maxHp")) or max(1.0, hp))
    effective_hp_ratio = min(2.0, (hp + block) / max_hp)

    target_index = _finite_number(params.get("enemyIndex"))
    enemy = (
        observation.enemy_by_index(int(target_index))
        if target_index is not None
        else None
    )
    incoming = enemy.incoming_attack if enemy is not None else 0.0
    return 20.0 * (1.0 - min(1.0, effective_hp_ratio)) + min(25.0, incoming * 0.75)


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


def _hand_card_for(
    action: JsonObject, observation: CombatObservation
) -> Mapping[str, Any]:
    params = _mapping(action.get("parameters"))
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
    if narrowed:
        matches = narrowed
    return _common_card_metadata(matches)


def _common_card_metadata(cards: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
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


def _potion_for(
    action: JsonObject, observation: CombatObservation
) -> Mapping[str, Any]:
    params = _mapping(action.get("parameters"))
    slot_number = _finite_number(params.get("potionSlot"))
    if slot_number is None:
        return {}
    slot = int(slot_number)
    if slot < 0 or slot >= len(observation.potions):
        return {}
    return observation.potions[slot]


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
    return number if math.isfinite(number) else None
