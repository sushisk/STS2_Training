"""Replaceable selection seam for post-combat card rewards.

The Emulator exposes card rewards as ``choice_reward_card`` actions at a
``reward_select`` boundary. This is intentionally separate from ``choice_card`` /
``pendingChoice.choiceSemantics`` combat-card selections.

The default policy preserves the historical Training behavior: choose uniformly from
all currently available reward actions (including skip/reroll/alternative actions when
they are present). Keeping that behavior behind a dedicated interface lets future card
quality, deck-context, or skip-threshold logic be added without changing the generic
combat selector again.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from sts2_training.selection.action_classification import JsonObject


class RewardCardSelectionPolicy(Protocol):
    """Policy interface for a decision containing ``choice_reward_card`` actions."""

    def select(
        self,
        legal_actions: Sequence[JsonObject],
        masked_emulator_dto: Mapping[str, Any],
        *,
        rng: random.Random,
    ) -> JsonObject:
        """Select one action from the reward decision's available legal actions."""


class RandomRewardCardSelectionPolicy:
    """Compatibility policy matching the pre-extraction reward-card behavior."""

    def select(
        self,
        legal_actions: Sequence[JsonObject],
        masked_emulator_dto: Mapping[str, Any],
        *,
        rng: random.Random,
    ) -> JsonObject:
        del masked_emulator_dto  # reserved for future deck/state-aware policies
        if not legal_actions:
            raise ValueError("reward card selection requires at least one legal action")
        return legal_actions[rng.randrange(len(legal_actions))]
