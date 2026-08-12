"""Evaluate ``choice_event_option`` candidates by simulating each option on a Branch.

Each candidate is assigned a distinct positive ``rng_id``. Root uses ``rng_id=0``;
event-evaluation Branches deliberately use non-root RNG hypotheses rather than treating
the game as deterministic. This is provisional: ideally each option would be sampled
across multiple RNG hypotheses, but that larger multi-sample change belongs in a
separate PR.

Whole Run implementation note: RL does not clone or retain a ``GameInstance`` per
public Branch. ``WholeRunInstance`` owns the public ``branch_id`` and Branch bookkeeping.
For a speculative action it sends ``WholeRunWorkerPool`` a replay recipe consisting of
the latest Map-boundary snapshot, room id, action prefix, target boundary, action id, and
(for Events) an ``EventRngReplayPlan``. A worker process owns one persistent
``WholeRunSession``/``GameInstance`` and reconstructs the requested frontier by loading
the Map snapshot, entering the room, replaying the prefix, applying the Event RNG
override at the original hypothesis point, and only then stepping the candidate action.
The returned result is converted back into the Branch view; descendant Branches are
reconstructed again from the same Map snapshot with an extended prefix. See
``STS2_RL/API/instance_whole_run.py`` and ``STS2_RL/Run/worker_pool.py``. This replay
architecture is also why a Whole Run Event before the first Map snapshot cannot be
branched.

``AsyncTrainingApiClient`` intentionally permits only one wire request in flight, so the
candidate simulations run sequentially here. Candidate-level faulted AND rejected
responses are treated as unusable samples - a rejection at this decision point is
expected whenever no Map snapshot exists yet (see the pre-Map Event note above; this is
literally every run's first decision), not evidence of a caller bug, so it must not sink
the whole run. Only a client left genuinely unusable (see ``_client_unusable``) still
propagates; transport/protocol failures propagate unconditionally because they can leave
the session's in-flight state ambiguous.

This is an HP-preservation heuristic, not a general Whole Run value function. Candidates
that fail to resolve, produce a non-finite HP value, or resolve to ``hp <= 0`` are ignored.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.api.contract import (
    ROOT_BRANCH_ID,
    RequestFaultedError,
    RequestRejectedError,
)
from sts2_training.api.transport import TransportError
from sts2_training.selection.action_classification import (
    JsonObject,
    choice_event_option_actions,
)

__all__ = ["best_event_option"]

_LOG = logging.getLogger(__name__)
_MAX_CLEANUP_RESERVE_S = 1.0


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
    ``None`` when there are fewer than two usable candidates, every candidate faults or
    resolves to death, or the evaluation budget expires before all candidates are tried.
    """

    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(float(timeout_s))
        or timeout_s <= 0
    ):
        raise ValueError("timeout_s must be a finite positive number")

    event_actions = choice_event_option_actions(legal_actions)
    if len(event_actions) <= 1:
        return None

    candidates: list[tuple[str, str]] = []
    for index, action in enumerate(event_actions):
        action_id = action.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            continue
        branch_id = f"event-eval-{decision_point_id}-{index}-{uuid.uuid4().hex[:8]}"
        candidates.append((branch_id, action_id))

    if len(candidates) <= 1:
        return None

    timeout_s = float(timeout_s)
    overall_deadline = time.monotonic() + timeout_s
    cleanup_reserve_s = min(_MAX_CLEANUP_RESERVE_S, timeout_s / (len(candidates) + 1))
    evaluation_deadline = overall_deadline - cleanup_reserve_s

    created_branch_ids: list[str] = []
    primary_error: BaseException | None = None
    try:
        best_action_id: str | None = None
        best_hp: float | None = None

        for index, (branch_id, action_id) in enumerate(candidates):
            remaining = evaluation_deadline - time.monotonic()
            if remaining <= 0:
                return None

            try:
                # ``branch_id`` is logical RL-coordinator state, not a cloned WorkerPool
                # GameInstance. The RL Whole Run path reconstructs this candidate on a
                # worker from Map snapshot + room + action prefix. ``rng_id`` selects the
                # Event RNG hypothesis whose replay plan is applied at the original Event
                # point before this action is stepped. See the module note above.
                result = await client.emulate_action(
                    instance_id,
                    ROOT_BRANCH_ID,
                    branch_id,
                    index + 1,
                    decision_point_id,
                    action_id,
                    timeout_s=remaining,
                )
            except (RequestFaultedError, RequestRejectedError):
                # Faulted requests consume the session sequence and may leave a faulted
                # Branch to release. A rejection here is most often the structural
                # pre-Map-snapshot case documented above, not a caller bug. Either way,
                # only a client left genuinely unusable is not a candidate-level failure
                # and must propagate.
                if _client_unusable(client):
                    raise
                created_branch_ids.append(branch_id)
                continue

            created_branch_ids.append(branch_id)
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
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        # release_branches cancels active Branches before releasing them. Reserve a small
        # part of the caller's deadline so a completed/faulted evaluation does not consume
        # the entire budget and make cleanup impossible.
        remaining = overall_deadline - time.monotonic()
        if created_branch_ids and remaining > 0 and not _client_unusable(client):
            try:
                await client.release_branches(
                    instance_id,
                    created_branch_ids,
                    timeout_s=remaining,
                )
            except RequestRejectedError:
                if primary_error is None and _client_unusable(client):
                    raise
                if primary_error is None:
                    _LOG.warning(
                        "event branch cleanup was rejected for instance_id=%s",
                        instance_id,
                        exc_info=True,
                    )
            except TransportError as exc:
                if primary_error is None and (
                    exc.completion_uncertain or _client_unusable(client)
                ):
                    raise
                if primary_error is None:
                    _LOG.warning(
                        "event branch cleanup transport failure for instance_id=%s",
                        instance_id,
                        exc_info=True,
                    )
            except Exception:
                if primary_error is None:
                    raise
                _LOG.exception(
                    "event branch cleanup also failed while propagating an evaluation "
                    "error for instance_id=%s",
                    instance_id,
                )


def _client_unusable(client: Any) -> bool:
    return getattr(client, "pending_retry", None) is not None or getattr(
        client, "session_invalid", False
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None
