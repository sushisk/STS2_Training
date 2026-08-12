"""Evaluate ``choice_event_option`` candidates by simulating each option on a Branch.

Each candidate is assigned a distinct positive ``rng_id``. Root uses ``rng_id=0``;
event-evaluation Branches deliberately use non-root RNG hypotheses rather than treating
the game as deterministic. This is provisional: ideally each option would be sampled
across multiple RNG hypotheses, but that larger multi-sample change belongs in a
separate PR.

Candidates are submitted together through ``emulate_actions``, the same batch transport
used by Combat frontier evaluation. The RL Whole Run implementation routes an Active
Event frontier through one WorkerPool dispatch while preserving each candidate's
``EventRngReplayPlan``.

This is an HP-preservation heuristic, not a general Whole Run value function. Candidates
that fail to resolve, produce a non-finite HP value, or resolve to ``hp <= 0`` are ignored.
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.api.contract import ROOT_BRANCH_ID
from sts2_training.selection.action_classification import (
    JsonObject,
    choice_event_option_actions,
)

__all__ = ["best_event_option"]


async def best_event_option(
    client: Any,
    *,
    instance_id: str,
    decision_point_id: str,
    legal_actions: Sequence[JsonObject],
    timeout_s: float,
) -> str | None:
    """Return the event option that leaves the player with the most HP.

    Each candidate is evaluated once under its own positive RNG hypothesis. Returns
    ``None`` when there are fewer than two usable candidates or every candidate fails,
    faults, or resolves to death.
    """

    event_actions = choice_event_option_actions(legal_actions)
    if len(event_actions) <= 1:
        return None

    branch_by_action: dict[str, str] = {}
    items: list[JsonObject] = []
    for index, action in enumerate(event_actions):
        action_id = action.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            continue
        branch_id = f"event-eval-{decision_point_id}-{index}-{uuid.uuid4().hex[:8]}"
        branch_by_action[branch_id] = action_id
        items.append(
            {
                "parent_branch_id": ROOT_BRANCH_ID,
                "branch_id": branch_id,
                "rng_id": index + 1,
                "decision_point_id": decision_point_id,
                "action_id": action_id,
            }
        )

    if len(items) <= 1:
        return None

    deadline = time.monotonic() + timeout_s
    created_branch_ids: list[str] = []
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        response = await client.emulate_actions(
            instance_id,
            items,
            timeout_s=remaining,
        )
        created_branch_ids.extend(branch_by_action)

        branch_results = response.get("branch_results")
        if not isinstance(branch_results, Mapping):
            return None

        best_action_id: str | None = None
        best_hp: float | None = None
        for branch_id, action_id in branch_by_action.items():
            result = branch_results.get(branch_id)
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
                best_action_id = action_id
        return best_action_id
    finally:
        # release_branches is defined to cancel first when needed, so a separate
        # cancel_branches call is redundant. Cleanup shares the caller's deadline.
        remaining = deadline - time.monotonic()
        if created_branch_ids and remaining > 0:
            try:
                await client.release_branches(
                    instance_id,
                    created_branch_ids,
                    timeout_s=remaining,
                )
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None
