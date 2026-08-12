"""Counterfactually evaluate ``choice_event_option`` actions on disposable Branches.

Every candidate is simulated from the same root decision with the same positive
``rng_id``. The wire contract defines that tuple as one shared RNG hypothesis, so HP
outcomes are comparable across candidate actions. If any candidate cannot be evaluated
cleanly, this module returns ``None`` and lets the caller use its normal heuristic
fallback instead of selecting from an incomplete comparison.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.api.contract import ROOT_BRANCH_ID
from sts2_training.selection.action_classification import (
    CHOICE_EVENT_OPTION_ACTION_TYPE,
    JsonObject,
)

__all__ = ["best_event_option"]

_EVENT_EVAL_RNG_ID = 1


async def best_event_option(
    client: Any,
    *,
    instance_id: str,
    decision_point_id: str,
    legal_actions: Sequence[JsonObject],
    timeout_s: float,
) -> "str | None":
    """Return the event option that preserves the most HP, or ``None`` on uncertainty.

    Dead outcomes (``hp <= 0``) are valid evaluations but are never selected. A failed,
    partial, malformed, or timed-out candidate makes the comparison incomplete, so the
    caller should fall back to its static event-choice heuristic.
    """

    event_actions = [
        action
        for action in legal_actions
        if action.get("action_type") == CHOICE_EVENT_OPTION_ACTION_TYPE
    ]
    if len(event_actions) <= 1:
        return None

    action_ids = [
        action_id
        for action in event_actions
        if isinstance((action_id := action.get("action_id")), str) and action_id
    ]
    if len(action_ids) != len(event_actions):
        return None

    deadline = time.monotonic() + timeout_s
    branch_ids: list[str] = []
    evaluated: list[tuple[str, float]] = []

    try:
        for index, action_id in enumerate(action_ids):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None

            branch_id = f"event-eval-{decision_point_id}-{index}-{uuid.uuid4().hex[:8]}"
            branch_ids.append(branch_id)
            try:
                result = await client.emulate_action(
                    instance_id,
                    ROOT_BRANCH_ID,
                    branch_id,
                    _EVENT_EVAL_RNG_ID,
                    decision_point_id,
                    action_id,
                    timeout_s=remaining,
                )
            except Exception:  # noqa: BLE001 - incomplete comparison falls back safely
                return None

            hp = _completed_hp(result)
            if hp is None:
                return None
            evaluated.append((action_id, hp))

        survivors = [(action_id, hp) for action_id, hp in evaluated if hp > 0]
        if not survivors:
            return None
        return max(survivors, key=lambda item: item[1])[0]
    finally:
        await _cleanup_branches(
            client,
            instance_id=instance_id,
            branch_ids=branch_ids,
            deadline=deadline,
        )


async def _cleanup_branches(
    client: Any,
    *,
    instance_id: str,
    branch_ids: Sequence[str],
    deadline: float,
) -> None:
    if not branch_ids:
        return

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return
    try:
        await client.cancel_branches(instance_id, branch_ids, timeout_s=remaining)
    except Exception:  # noqa: BLE001 - cleanup is best-effort
        pass

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return
    try:
        await client.release_branches(instance_id, branch_ids, timeout_s=remaining)
    except Exception:  # noqa: BLE001 - cleanup is best-effort
        pass


def _completed_hp(result: Any) -> "float | None":
    if not isinstance(result, Mapping) or result.get("status") != "completed":
        return None
    result_dto = result.get("masked_emulator_dto")
    if not isinstance(result_dto, Mapping):
        return None
    return _finite_number(result_dto.get("hp"))


def _finite_number(value: Any) -> "float | None":
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None
