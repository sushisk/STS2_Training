"""Branch-evaluate ``choice_event_option`` decisions via Whole Run's ``emulate_action``
(the "Active Event RNG Hypothesis" machinery - see
``Outputs/reports/rl_active_event_rng_hypothesis_20260804.md``). That machinery was
built to compare different RNG resolutions of the *same* chosen option, but the
Emulator/RL side never distinguishes that use from comparing *different* candidate
options against each other - ``emulate_action``'s ``action_id`` names whichever legal
action a caller picks. This module repurposes it for the latter.

Uses the single-item ``emulate_action`` call, not the batch ``emulate_actions`` -
confirmed live that Whole Run's batch endpoint rejects Branch simulation at the
``event_choice`` boundary ("Branch simulation is unavailable at this Whole Run
boundary.") even though the single-item call is explicitly supported there (see
``instance_whole_run.py::emulate_action``'s own boundary check). Candidates are
evaluated concurrently via ``asyncio.gather``, mirroring ``test_e2e.py``'s own sibling
Branch pattern (multiple ``emulate_action`` calls from the same root decision point).

Why this exists: ``choice_event_option`` previously had only a static, predictive hard
safety filter (``event_choice_heuristic.safe_event_option_candidates``, based on the
Emulator's own pre-computed ``willKillPlayer`` flag) - it never weighed HP cost/benefit
beyond that binary "definitely lethal" cutoff, so e.g. an event offering one free option
and one option that costs meaningful-but-survivable HP was chosen from uniformly at
random. This module instead actually resolves every candidate on a disposable Branch and
compares the real resulting HP.

Scope: HP preservation only, not a general Whole Run value function. ``ValueModel``
(``decision/value.py``) is explicitly Combat-domain-scoped (``CombatObservation.from_dto``
requires combat-shaped fields an event's resulting ``map_select`` DTO doesn't have), so it
cannot be reused here without a new, much larger Whole-Run value model - out of scope for
the concrete problem this fixes (event choices spending HP for no evaluated reason).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.api.contract import ROOT_BRANCH_ID
from sts2_training.selection.action_classification import (
    CHOICE_EVENT_OPTION_ACTION_TYPE,
    JsonObject,
)

__all__ = ["best_event_option"]


async def best_event_option(
    client: Any,
    *,
    instance_id: str,
    decision_point_id: str,
    legal_actions: Sequence[JsonObject],
    timeout_s: float,
) -> "str | None":
    """Branch-evaluate every available ``choice_event_option`` candidate and return the
    ``action_id`` of whichever leaves the player with the most HP, excluding any
    candidate that resolves to the player being dead (``hp <= 0``) or to an unusable
    branch result.

    Returns ``None`` - meaning "no opinion, caller should fall back to its own logic
    (beam search / ``event_choice_heuristic``'s static lethality filter)" - when there
    are 0-1 candidates (nothing to compare) or every branch failed/errored.
    """

    event_actions = [
        action
        for action in legal_actions
        if action.get("action_type") == CHOICE_EVENT_OPTION_ACTION_TYPE
    ]
    if len(event_actions) <= 1:
        return None

    branch_by_action: dict[str, str] = {}
    for index, action in enumerate(event_actions):
        action_id = action.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            continue
        branch_id = f"event-eval-{decision_point_id}-{index}-{uuid.uuid4().hex[:8]}"
        branch_by_action[branch_id] = action_id

    if len(branch_by_action) <= 1:
        return None

    branch_ids = list(branch_by_action)
    try:
        results = await asyncio.gather(
            *(
                _emulate_one(
                    client,
                    instance_id=instance_id,
                    branch_id=branch_id,
                    rng_id=index + 1,
                    decision_point_id=decision_point_id,
                    action_id=branch_by_action[branch_id],
                    timeout_s=timeout_s,
                )
                for index, branch_id in enumerate(branch_ids)
            ),
            return_exceptions=True,
        )

        best_action_id: "str | None" = None
        best_hp: "float | None" = None
        for branch_id, result in zip(branch_ids, results):
            if isinstance(result, BaseException) or not isinstance(result, Mapping):
                continue
            if result.get("status") != "completed":
                continue
            result_dto = result.get("masked_emulator_dto")
            if not isinstance(result_dto, Mapping):
                continue
            hp = _finite_number(result_dto.get("hp"))
            if hp is None or hp <= 0:
                continue
            if best_hp is None or hp > best_hp:
                best_hp = hp
                best_action_id = branch_by_action[branch_id]
        return best_action_id
    finally:
        try:
            await client.cancel_branches(instance_id, branch_ids, timeout_s=timeout_s)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
        try:
            await client.release_branches(instance_id, branch_ids, timeout_s=timeout_s)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass


async def _emulate_one(
    client: Any,
    *,
    instance_id: str,
    branch_id: str,
    rng_id: int,
    decision_point_id: str,
    action_id: str,
    timeout_s: float,
) -> "JsonObject | None":
    try:
        return await client.emulate_action(
            instance_id,
            ROOT_BRANCH_ID,
            branch_id,
            rng_id,
            decision_point_id,
            action_id,
            timeout_s=timeout_s,
        )
    except Exception:  # noqa: BLE001 - one candidate's failure must not sink the rest
        return None


def _finite_number(value: Any) -> "float | None":
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None
