"""Selection policies for post-combat card rewards."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from sts2_training.selection.action_classification import (
    CHOICE_REWARD_CARD_ACTION_TYPE,
    JsonObject,
    reward_card_actions,
)

_CARD_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "external_data" / "cards_all.json"


class RewardCardSelectionPolicy(Protocol):
    """Policy for decisions containing ``choice_reward_card`` actions."""

    def select(
        self,
        legal_actions: Sequence[JsonObject],
        masked_emulator_dto: Mapping[str, Any],
        *,
        rng: random.Random,
    ) -> JsonObject:
        """Select one of the available reward actions."""


class RandomRewardCardSelectionPolicy:
    """Choose uniformly from all available reward actions."""

    def select(
        self,
        legal_actions: Sequence[JsonObject],
        masked_emulator_dto: Mapping[str, Any],
        *,
        rng: random.Random,
    ) -> JsonObject:
        del masked_emulator_dto
        if not legal_actions:
            raise ValueError("reward card selection requires at least one legal action")
        return legal_actions[rng.randrange(len(legal_actions))]


@lru_cache(maxsize=1)
def _card_scores() -> dict[str, float]:
    """``card_id`` -> ``skada_score``, loaded from the sts2log.com card stats export.

    ``skada_score`` is used as-is as a single composite card-quality prior (observed
    range ~908-1054 across all 439 cards) rather than combining it with the export's
    other win-rate fields, since a single well-attested field is simpler to reason
    about than a hand-tuned blend of several correlated ones.
    """
    with _CARD_DATA_PATH.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    scores: dict[str, float] = {}
    for card in payload.get("cards", []):
        card_id = card.get("card_id")
        score = card.get("skada_score")
        if isinstance(card_id, str) and isinstance(score, (int, float)):
            scores[card_id] = float(score)
    return scores


class CardDataRewardCardSelectionPolicy:
    """Choose the offered card with the highest known ``skada_score``.

    Only scores ``choice_reward_card`` candidates - skip/reroll/alternative actions
    carry no comparable quality signal in the underlying data (see `_card_scores`), so
    this policy always takes a card when one is offered. A card missing from the data
    (should not happen for the current character pools) falls back to indifference
    (score 0.0) rather than failing the decision.
    """

    def __init__(self, card_scores: Mapping[str, float] | None = None) -> None:
        self._card_scores = card_scores if card_scores is not None else _card_scores()

    def select(
        self,
        legal_actions: Sequence[JsonObject],
        masked_emulator_dto: Mapping[str, Any],
        *,
        rng: random.Random,
    ) -> JsonObject:
        del masked_emulator_dto
        candidates = reward_card_actions(legal_actions)
        if not candidates:
            if not legal_actions:
                raise ValueError("reward card selection requires at least one legal action")
            return legal_actions[rng.randrange(len(legal_actions))]

        best_score = max(self._score(action) for action in candidates)
        best_candidates = [
            action for action in candidates if self._score(action) == best_score
        ]
        return best_candidates[rng.randrange(len(best_candidates))]

    def _score(self, action: JsonObject) -> float:
        params = action.get("parameters")
        card_id = params.get("cardId") if isinstance(params, Mapping) else None
        if not isinstance(card_id, str):
            return 0.0
        return self._card_scores.get(card_id, 0.0)
