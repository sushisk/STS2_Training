"""Evaluate ``choice_event_option`` candidates by simulating each option on a Branch.

Every candidate uses the same positive ``rng_id`` so the alternatives are compared under
one RNG hypothesis for the same root decision. The single-item ``emulate_action`` calls
are scheduled together with ``asyncio.gather``; when this runs through
``AsyncTrainingApiClient`` the wire requests are still serialized by that client's
operation lock. Actual parallel Branch evaluation is intentionally outside this module.

This is an HP-preservation heuristic, not a general Whole Run value function. Candidates
that fail to resolve, produce a non-finite HP value, or resolve to ``hp <= 0`` are ignored.
"""

from __future__ import annotations

import asyncio
import math
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.api.contract import ROOT_BRANCH_ID
from sts2_training.selection.action_classification import (
    JsonObject,
    choice_event_option_actions,
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
) -> str | None:
    """Return the event option that leaves the player with the most HP.

    All candidates are evaluated under the same RNG hypothesis. Returns ``None`` when
    there are fewer than two usable candidates or every candidate fails, faults, or
    resolves to death.
    """

    event_actions = choice_event_option_actions(legal_actions)
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
                    rng_id=_EVENT_EVAL_RNG_ID,
                    decision_point_id=decision_point_id,
                    action_id=branch_by_action[branch_id],
                    timeout_s=timeout_s,
                )
                for branch_id in branch_ids
            )
        )

        best_action_id: str | None = None
        best_hp: float | None = None
        for branch_id, result in zip(branch_ids, results):
            if not isinstance(result, Mapping) or result.get("status") != "completed":
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
) -> JsonObject | None:
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


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None
