"""Evaluate ``choice_event_option`` candidates by simulating each option on a Branch.

Each candidate is assigned a distinct positive ``rng_id``. Root uses ``rng_id=0``;
event-evaluation Branches deliberately use non-root RNG hypotheses rather than treating
the game as deterministic. This is provisional: ideally each option would be sampled
across multiple RNG hypotheses, but that larger multi-sample change belongs in a
separate PR.

When RL advertises ``event_choice`` in ``emulate_actions_boundaries`` candidates are
submitted together through ``emulate_actions``. Older RL deployments omit that semantic
capability, so Training falls back to the existing singleton ``emulate_action`` route
instead of relying on cross-repository deployment order.

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

A usable request-level rejection at this decision point is treated as an unavailable
sample rather than a fatal run error. In particular, the pre-Map Event that starts a run
cannot be branched even on an RL server that advertises Event batch support. A rejected
batch therefore yields no Event score, while singleton fallback skips a rejected
candidate. Faulted requests are also unusable samples, but may have created Branch
state and are cleaned up. Transport/protocol failures propagate; completion-uncertain
requests transfer Branch ownership to their exact retry token so cleanup cannot race the
replay.

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
_EVENT_BATCH_BOUNDARY = "event_choice"


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
    is rejected, resolves to death, or the evaluation budget expires before all
    candidates are tried.

    Branch ownership stays with this lifecycle even across an exact transport replay.
    If an evaluation becomes completion-uncertain, the client is told which Branches
    this search owns; once the exact retry resolves, the client releases those Branches
    before allowing the recovered request to escape back to its caller.
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

    candidates: list[tuple[str, str, int]] = []
    for index, action in enumerate(event_actions):
        action_id = action.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            continue
        branch_id = f"event-eval-{decision_point_id}-{index}-{uuid.uuid4().hex[:8]}"
        candidates.append((branch_id, action_id, index + 1))

    if len(candidates) <= 1:
        return None

    timeout_s = float(timeout_s)
    overall_deadline = time.monotonic() + timeout_s
    cleanup_reserve_s = min(_MAX_CLEANUP_RESERVE_S, timeout_s / (len(candidates) + 1))
    evaluation_deadline = overall_deadline - cleanup_reserve_s

    created_branch_ids: list[str] = []
    primary_error: BaseException | None = None
    try:
        if _supports_event_batch(client):
            remaining = evaluation_deadline - time.monotonic()
            if remaining <= 0:
                return None
            items = [
                {
                    "parent_branch_id": ROOT_BRANCH_ID,
                    "branch_id": branch_id,
                    "rng_id": rng_id,
                    "decision_point_id": decision_point_id,
                    "action_id": action_id,
                }
                for branch_id, action_id, rng_id in candidates
            ]
            try:
                response = await client.emulate_actions(
                    instance_id,
                    items,
                    timeout_s=remaining,
                )
            except RequestRejectedError:
                # Validation rejection means the batch did not create its candidate
                # Branches. The common pre-Map Event is expected to take this path.
                if _client_unusable(client):
                    raise
                return None
            except RequestFaultedError:
                # A fault consumes the request and may leave any/all candidate Branches
                # behind, so own them until the common release in ``finally``.
                if _client_unusable(client):
                    raise
                created_branch_ids.extend(
                    branch_id for branch_id, _, _ in candidates
                )
                return None
            except BaseException:
                # Completion uncertainty applies to the whole batch: any/all Branches
                # may already exist server-side. Preserve their ownership on the exact
                # replay token rather than attempting a different cleanup request.
                _defer_retry_cleanup(
                    client,
                    instance_id=instance_id,
                    branch_ids=[branch_id for branch_id, _, _ in candidates],
                )
                raise

            created_branch_ids.extend(branch_id for branch_id, _, _ in candidates)
            branch_results = response.get("branch_results")
            if not isinstance(branch_results, Mapping):
                return None
            return _best_from_batch(branch_results, candidates)

        best_action_id: str | None = None
        best_hp: float | None = None
        for branch_id, action_id, rng_id in candidates:
            remaining = evaluation_deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                result = await client.emulate_action(
                    instance_id,
                    ROOT_BRANCH_ID,
                    branch_id,
                    rng_id,
                    decision_point_id,
                    action_id,
                    timeout_s=remaining,
                )
            except RequestRejectedError:
                # A usable rejection is an unavailable candidate (most notably the
                # structural pre-Map case) and did not create this Branch.
                if _client_unusable(client):
                    raise
                continue
            except RequestFaultedError:
                if _client_unusable(client):
                    raise
                created_branch_ids.append(branch_id)
                continue
            except BaseException:
                # Earlier singleton candidates are already owned by this scope and the
                # current candidate may have completed server-side. Carry all of them
                # with the exact retry so recovery can release the complete set.
                _defer_retry_cleanup(
                    client,
                    instance_id=instance_id,
                    branch_ids=[*created_branch_ids, branch_id],
                )
                raise

            created_branch_ids.append(branch_id)
            hp = _hp_from_result(result)
            if hp is not None and (best_hp is None or hp > best_hp):
                best_hp = hp
                best_action_id = action_id

        return best_action_id
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        # release_branches cancels active Branches before releasing them. Never send a
        # different request while exact replay is pending; deferred cleanup is performed
        # by AsyncTrainingApiClient after that replay resolves.
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


def _supports_event_batch(client: Any) -> bool:
    boundaries = getattr(client, "emulate_actions_boundaries", ())
    return isinstance(boundaries, (set, frozenset, list, tuple)) and (
        _EVENT_BATCH_BOUNDARY in boundaries
    )


def _defer_retry_cleanup(
    client: Any,
    *,
    instance_id: str,
    branch_ids: Sequence[str],
) -> None:
    retry = getattr(client, "pending_retry", None)
    defer = getattr(client, "defer_branch_cleanup_after_retry", None)
    if retry is None or not callable(defer):
        return
    defer(retry, instance_id, branch_ids)


def _best_from_batch(
    branch_results: Mapping[str, Any],
    candidates: Sequence[tuple[str, str, int]],
) -> str | None:
    best_action_id: str | None = None
    best_hp: float | None = None
    for branch_id, action_id, _ in candidates:
        result = branch_results.get(branch_id)
        hp = _hp_from_result(result)
        if hp is not None and (best_hp is None or hp > best_hp):
            best_hp = hp
            best_action_id = action_id
    return best_action_id


def _hp_from_result(result: Any) -> float | None:
    if not isinstance(result, Mapping) or result.get("status") != "completed":
        return None
    result_dto = result.get("masked_emulator_dto")
    if not isinstance(result_dto, Mapping):
        return None
    hp = _finite_number(result_dto.get("hp"))
    return hp if hp is not None and hp > 0 else None


def _client_unusable(client: Any) -> bool:
    return getattr(client, "pending_retry", None) is not None or getattr(
        client, "session_invalid", False
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None
