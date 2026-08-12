"""Evaluate ``choice_event_option`` candidates by simulating each option on a Branch.

Each candidate is assigned a distinct positive ``rng_id``. Root uses ``rng_id=0``;
event-evaluation Branches deliberately use non-root RNG hypotheses rather than treating
the game as deterministic. This is provisional: ideally each option would be sampled
across multiple RNG hypotheses, but that larger multi-sample change belongs in a
separate PR.

``AsyncTrainingApiClient`` intentionally permits only one wire request in flight, so the
candidate simulations run sequentially here. Candidate-level API faults are treated as
unusable samples; transport/protocol failures propagate because they can make the client
session unsafe to continue.

This is an HP-preservation heuristic, not a general Whole Run value function. Candidates
that fail to resolve, produce a non-finite HP value, or resolve to ``hp <= 0`` are ignored.
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.api.contract import ApiOperationError, ROOT_BRANCH_ID
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
    for index, action in enumerate(event_actions):
        action_id = action.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            continue
        branch_id = f"event-eval-{decision_point_id}-{index}-{uuid.uuid4().hex[:8]}"
        branch_by_action[branch_id] = action_id

    if len(branch_by_action) <= 1:
        return None

    deadline = time.monotonic() + timeout_s
    created_branch_ids: list[str] = []
    results: list[tuple[str, JsonObject | None]] = []
    try:
        for index, (branch_id, action_id) in enumerate(branch_by_action.items()):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None

            try:
                result = await client.emulate_action(
                    instance_id,
                    ROOT_BRANCH_ID,
                    branch_id,
                    index + 1,
                    decision_point_id,
                    action_id,
                    timeout_s=remaining,
                )
            except ApiOperationError as exc:
                # A faulted request was admitted and may have created a Branch; a
                # rejected request did not mutate state. Session-fatal rejections must
                # propagate rather than being disguised as a bad candidate sample.
                if getattr(client, "session_invalid", False):
                    raise
                if exc.response.get("status") == "faulted":
                    created_branch_ids.append(branch_id)
                results.append((action_id, None))
                continue

            created_branch_ids.append(branch_id)
            results.append((action_id, result))

        best_action_id: str | None = None
        best_hp: float | None = None
        for action_id, result in results:
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
