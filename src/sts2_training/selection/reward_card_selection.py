"""Selection policies for post-combat card rewards."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from sts2_training.selection.action_classification import JsonObject


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
