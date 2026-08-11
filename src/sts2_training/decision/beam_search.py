"""`BeamSearchEngine`: policy-guided beam search over `AsyncTrainingApiClient`.

One logical frontier is sent as one or more bounded `emulate_actions` batches. Combat
lookahead depth counts ordinary system/card/potion actions. Interactive continuation
steps (`choice_target`, `choice_card`, `choice_confirm`, `choice_skip`) are tracked by a
separate bounded counter so a card or potion can be followed through its pending choice
even when the ordinary combat-depth budget has been reached.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from sts2_training.api.contract import ROOT_BRANCH_ID, RequestRejectedError
from sts2_training.api.transport import TransportError
from sts2_training.decision.policy import ActionCandidate, PolicyModel
from sts2_training.decision.value import ValueModel

JsonObject = dict[str, Any]

_LOG = logging.getLogger(__name__)
_RESOLVED_STATUSES = frozenset({"completed", "partial"})
_CONTINUATION_ACTION_TYPES = frozenset(
    {"choice_target", "choice_card", "choice_confirm", "choice_skip"}
)


@dataclass
class BeamSearchConfig:
    beam_width: int = 8
    top_k_actions: int = 4
    max_depth: int = 2
    simulation_options: Mapping[str, Any] | None = None
    time_budget_ms: float | None = None
    max_batch_size: int = 64
    expand_partial: bool = True
    release_branches_on_finish: bool = True
    beam_searchable_action_types: frozenset[str] = field(
        default_factory=lambda: frozenset({"system", "card", "potion"})
    )
    max_continuation_steps: int = 8

    def __post_init__(self) -> None:
        for name in (
            "beam_width",
            "top_k_actions",
            "max_depth",
            "max_batch_size",
            "max_continuation_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.time_budget_ms is not None and (
            isinstance(self.time_budget_ms, bool)
            or not isinstance(self.time_budget_ms, (int, float))
            or not math.isfinite(float(self.time_budget_ms))
            or self.time_budget_ms <= 0
        ):
            raise ValueError("time_budget_ms must be a finite positive number when provided")
        if self.simulation_options is None:
            self.simulation_options = {"stop_condition": "next_decision"}


@dataclass(frozen=True)
class BeamNode:
    branch_id: str
    parent_branch_id: str
    rng_id: int
    decision_point_id: str
    masked_emulator_dto: Mapping[str, Any]
    depth: int
    value: float
    root_action_id: str | None
    combat_depth: int = 0
    continuation_steps: int = 0
    branch_log: tuple[Any, ...] = ()
    terminal: bool = False


@dataclass
class BeamSearchStats:
    depths_completed: int = 0
    nodes_expanded: int = 0
    branches_created: int = 0
    policy_ms: float = 0.0
    emulate_actions_ms: float = 0.0
    value_ms: float = 0.0
    cleanup_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(frozen=True)
class BeamSearchResult:
    best_root_action_id: str | None
    best_value: float | None
    best_node: BeamNode | None
    reason: str
    stats: BeamSearchStats


class BranchIdAllocator:
    """Issues branch_id/rng_id values unique for as long as this allocator lives."""

    def __init__(self, prefix: str | None = None) -> None:
        self._prefix = prefix or uuid.uuid4().hex[:8]
        self._branch_counter = 0
        self._rng_counter = 0

    def next_branch_id(self) -> str:
        self._branch_counter += 1
        return f"bs-{self._prefix}-{self._branch_counter}"

    def next_rng_id(self) -> int:
        self._rng_counter += 1
        return self._rng_counter


class BeamSearchEngine:
    """Construct once per `AsyncTrainingApiClient` instance/session and reuse."""

    def __init__(
        self,
        client: Any,
        *,
        policy: PolicyModel,
        value_fn: ValueModel,
        config: BeamSearchConfig | None = None,
    ) -> None:
        self._client = client
        self._policy = policy
        self._value_fn = value_fn
        self.config = config or BeamSearchConfig()
        self._allocator = BranchIdAllocator()

    async def search(
        self,
        instance_id: str,
        root_decision: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> BeamSearchResult:
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(float(timeout_s))
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be a finite positive number")

        cfg = self.config
        overall_deadline = time.monotonic() + timeout_s
        search_deadline = overall_deadline
        if cfg.time_budget_ms is not None:
            search_deadline = min(
                search_deadline, time.monotonic() + cfg.time_budget_ms / 1000.0
            )

        stats = BeamSearchStats()
        t_start = time.monotonic()

        root_dto = root_decision.get("masked_emulator_dto")
        root_decision_point_id = root_decision.get("decision_point_id")
        if not isinstance(root_dto, Mapping) or not isinstance(root_decision_point_id, str):
            return BeamSearchResult(None, None, None, "invalid_root_decision", stats)

        root_legal_actions = root_dto.get("legal_actions")
        if not root_legal_actions:
            return BeamSearchResult(None, None, None, "no_legal_actions", stats)
        if getattr(self._client, "instance_type", None) == "whole_run":
            return BeamSearchResult(None, None, None, "emulate_actions_not_supported", stats)
        if not _is_beam_searchable(root_dto, cfg.beam_searchable_action_types):
            return BeamSearchResult(None, None, None, "not_beam_searchable", stats)

        root_node = BeamNode(
            branch_id=ROOT_BRANCH_ID,
            parent_branch_id=ROOT_BRANCH_ID,
            rng_id=0,
            decision_point_id=root_decision_point_id,
            masked_emulator_dto=root_dto,
            depth=0,
            value=0.0,
            root_action_id=None,
        )

        beam: list[BeamNode] = [root_node]
        finished: list[BeamNode] = []
        all_branch_ids: list[str] = []
        reason = "max_depth"
        search_error: BaseException | None = None

        try:
            while beam:
                if time.monotonic() >= search_deadline:
                    reason = "time_budget"
                    break

                searchable: list[BeamNode] = []
                for node in beam:
                    if _is_beam_searchable(
                        node.masked_emulator_dto, cfg.beam_searchable_action_types
                    ):
                        searchable.append(node)
                    else:
                        finished.append(node)
                beam = searchable
                if not beam:
                    reason = "not_beam_searchable"
                    break

                items, item_meta, policy_ms = self._propose_frontier(beam)
                stats.policy_ms += policy_ms
                if not items:
                    reason = "no_candidates"
                    break

                branch_results, fatal_reason = await self._emulate_depth_batch(
                    instance_id, items, item_meta, all_branch_ids, stats, search_deadline
                )

                (
                    next_beam,
                    newly_finished,
                    value_ms,
                    hit_max_depth,
                    hit_continuation_limit,
                ) = self._score_frontier(item_meta, branch_results)
                stats.value_ms += value_ms
                stats.nodes_expanded += len(branch_results)
                if branch_results and not next_beam and not newly_finished:
                    raise RuntimeError("all emulate_actions branch results faulted")
                finished.extend(newly_finished)

                next_beam.sort(key=lambda n: n.value, reverse=True)
                beam = next_beam[: cfg.beam_width]

                if fatal_reason is not None:
                    reason = fatal_reason
                    break
                stats.depths_completed += 1

                if not beam:
                    if hit_continuation_limit:
                        reason = "max_continuation_steps"
                    elif hit_max_depth:
                        reason = "max_depth"
                    else:
                        reason = "beam_exhausted"
                    break

            finished.extend(beam)
        except BaseException as exc:
            search_error = exc
            raise
        finally:
            t0 = time.monotonic()
            try:
                await self._cleanup(instance_id, all_branch_ids, deadline=overall_deadline)
            except Exception:
                if search_error is None:
                    raise
                _LOG.exception(
                    "beam search cleanup also failed while propagating a search error "
                    "for instance_id=%s",
                    instance_id,
                )
            finally:
                stats.cleanup_ms += (time.monotonic() - t0) * 1000.0

        stats.total_ms = (time.monotonic() - t_start) * 1000.0

        actionable = [node for node in finished if node.root_action_id is not None]
        if not actionable:
            return BeamSearchResult(None, None, None, reason, stats)
        best_node = max(actionable, key=lambda n: n.value)
        return BeamSearchResult(
            best_node.root_action_id, best_node.value, best_node, reason, stats
        )

    def _propose_frontier(
        self, beam: Sequence[BeamNode]
    ) -> tuple[list[JsonObject], list[tuple[BeamNode, ActionCandidate, str, int]], float]:
        t0 = time.monotonic()
        requests = [
            (node.masked_emulator_dto.get("legal_actions") or [], node.masked_emulator_dto)
            for node in beam
        ]
        proposals = self._policy.propose_batch(requests, top_k=self.config.top_k_actions)
        policy_ms = (time.monotonic() - t0) * 1000.0
        if len(proposals) != len(beam):
            raise RuntimeError("PolicyModel.propose_batch must return exactly one entry per request")

        items: list[JsonObject] = []
        item_meta: list[tuple[BeamNode, ActionCandidate, str, int]] = []
        for node, candidates in zip(beam, proposals):
            if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
                raise RuntimeError("PolicyModel.propose_batch entries must be candidate sequences")
            if len(candidates) > self.config.top_k_actions:
                raise RuntimeError("PolicyModel.propose_batch returned more than top_k candidates")

            available_ids = _available_action_ids(node.masked_emulator_dto)
            for candidate in candidates:
                if not isinstance(candidate, ActionCandidate):
                    raise RuntimeError("PolicyModel.propose_batch must return ActionCandidate objects")
                if candidate.action_id not in available_ids:
                    raise RuntimeError(
                        "PolicyModel proposed an action_id that is not currently available: "
                        f"{candidate.action_id!r}"
                    )
                branch_id = self._allocator.next_branch_id()
                rng_id = (
                    node.rng_id if node.branch_id != ROOT_BRANCH_ID else self._allocator.next_rng_id()
                )
                items.append(
                    {
                        "parent_branch_id": node.branch_id,
                        "branch_id": branch_id,
                        "rng_id": rng_id,
                        "decision_point_id": node.decision_point_id,
                        "action_id": candidate.action_id,
                    }
                )
                item_meta.append((node, candidate, branch_id, rng_id))
        return items, item_meta, policy_ms

    async def _emulate_depth_batch(
        self,
        instance_id: str,
        items: Sequence[JsonObject],
        item_meta: Sequence[tuple[BeamNode, ActionCandidate, str, int]],
        all_branch_ids: list[str],
        stats: BeamSearchStats,
        deadline: float,
    ) -> tuple[dict[str, Any], str | None]:
        cfg = self.config
        batch_size = cfg.max_batch_size
        server_limit = getattr(self._client, "max_emulate_actions_items", None)
        if isinstance(server_limit, int) and not isinstance(server_limit, bool) and server_limit > 0:
            batch_size = min(batch_size, server_limit)

        branch_results: dict[str, Any] = {}
        for start in range(0, len(items), batch_size):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return branch_results, "time_budget"
            chunk_items = items[start : start + batch_size]
            chunk_meta = item_meta[start : start + batch_size]

            t0 = time.monotonic()
            try:
                response = await self._client.emulate_actions(
                    instance_id,
                    chunk_items,
                    timeout_s=remaining,
                    simulation_options=cfg.simulation_options,
                )
            except RequestRejectedError as exc:
                stats.emulate_actions_ms += (time.monotonic() - t0) * 1000.0
                if _client_unusable(self._client):
                    raise
                fault_kind = exc.response.get("fault_kind")
                detail = (
                    fault_kind
                    if isinstance(fault_kind, str) and fault_kind
                    else type(exc).__name__
                )
                return branch_results, f"emulate_actions_rejected:{detail}"
            stats.emulate_actions_ms += (time.monotonic() - t0) * 1000.0

            all_branch_ids.extend(meta[2] for meta in chunk_meta)
            stats.branches_created += len(chunk_meta)
            branch_results.update(response.get("branch_results") or {})
        return branch_results, None

    def _score_frontier(
        self,
        item_meta: Sequence[tuple[BeamNode, ActionCandidate, str, int]],
        branch_results: Mapping[str, Any],
        depth: int | None = None,
    ) -> tuple[list[BeamNode], list[BeamNode], float, bool, bool]:
        cfg = self.config
        resolved: list[
            tuple[
                BeamNode,
                ActionCandidate,
                str,
                int,
                Mapping[str, Any],
                Mapping[str, Any],
                Any,
            ]
        ] = []
        for node, candidate, branch_id, rng_id in item_meta:
            result = branch_results.get(branch_id)
            if not isinstance(result, Mapping) or result.get("status") not in _RESOLVED_STATUSES:
                continue
            dto = result.get("masked_emulator_dto")
            if not isinstance(dto, Mapping):
                continue
            resolved.append((node, candidate, branch_id, rng_id, result, dto, result.get("status")))

        t0 = time.monotonic()
        raw_values = self._value_fn.evaluate_batch([entry[5] for entry in resolved]) if resolved else []
        value_ms = (time.monotonic() - t0) * 1000.0
        values = _validated_values(raw_values, expected=len(resolved))

        next_beam: list[BeamNode] = []
        newly_finished: list[BeamNode] = []
        hit_max_depth = False
        hit_continuation_limit = False

        for (node, candidate, branch_id, rng_id, result, dto, status), value in zip(
            resolved, values
        ):
            action_type = _action_type_for_id(node.masked_emulator_dto, candidate.action_id)
            if action_type is None:
                raise RuntimeError(
                    f"selected action_id {candidate.action_id!r} has no valid action_type"
                )
            is_continuation_action = action_type in _CONTINUATION_ACTION_TYPES
            combat_depth = node.combat_depth + (0 if is_continuation_action else 1)
            continuation_steps = (
                node.continuation_steps + 1 if is_continuation_action else 0
            )

            decision_point_id = result.get("decision_point_id")
            new_node = BeamNode(
                branch_id=branch_id,
                parent_branch_id=node.branch_id,
                rng_id=rng_id,
                decision_point_id=decision_point_id if isinstance(decision_point_id, str) else "",
                masked_emulator_dto=dto,
                depth=node.depth + 1,
                value=value,
                root_action_id=node.root_action_id or candidate.action_id,
                combat_depth=combat_depth,
                continuation_steps=continuation_steps,
                branch_log=tuple(result.get("branch_log") or ()),
                terminal=_is_terminal(dto),
            )
            cannot_expand = not new_node.decision_point_id or (
                status == "partial" and not cfg.expand_partial
            )
            if new_node.terminal or cannot_expand:
                newly_finished.append(new_node)
                continue

            if _is_continuation_decision(dto):
                if new_node.continuation_steps >= cfg.max_continuation_steps:
                    hit_continuation_limit = True
                    newly_finished.append(new_node)
                else:
                    next_beam.append(new_node)
                continue

            if new_node.combat_depth >= cfg.max_depth:
                hit_max_depth = True
                newly_finished.append(new_node)
            else:
                next_beam.append(new_node)

        return (
            next_beam,
            newly_finished,
            value_ms,
            hit_max_depth,
            hit_continuation_limit,
        )

    async def _cleanup(
        self,
        instance_id: str,
        branch_ids: list[str],
        *,
        deadline: float,
    ) -> None:
        if not branch_ids or not self.config.release_branches_on_finish:
            return
        if _client_unusable(self._client):
            _LOG.warning(
                "skipping beam search branch cleanup for instance_id=%s: client has "
                "pending_retry or an invalid session",
                instance_id,
            )
            return

        cancelled = await self._cleanup_call(
            "cancel_branches", instance_id, branch_ids, deadline
        )
        if not cancelled:
            _LOG.warning(
                "skipping beam search branch release for instance_id=%s because "
                "cancellation did not complete",
                instance_id,
            )
            return
        if _client_unusable(self._client):
            _LOG.warning(
                "skipping beam search branch release for instance_id=%s after cancellation failure",
                instance_id,
            )
            return
        await self._cleanup_call("release_branches", instance_id, branch_ids, deadline)

    async def _cleanup_call(
        self,
        operation: str,
        instance_id: str,
        branch_ids: list[str],
        deadline: float,
    ) -> bool:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _LOG.warning(
                "skipping beam search %s for instance_id=%s: timeout budget exhausted",
                operation,
                instance_id,
            )
            return False

        try:
            await getattr(self._client, operation)(
                instance_id, branch_ids, timeout_s=min(5.0, remaining)
            )
        except RequestRejectedError:
            if _client_unusable(self._client):
                raise
            _LOG.warning(
                "beam search %s was rejected for instance_id=%s (%d branches)",
                operation,
                instance_id,
                len(branch_ids),
                exc_info=True,
            )
            return False
        except TransportError as exc:
            if exc.completion_uncertain or _client_unusable(self._client):
                raise
            _LOG.warning(
                "beam search %s transport failure for instance_id=%s (%d branches)",
                operation,
                instance_id,
                len(branch_ids),
                exc_info=True,
            )
            return False
        return True


def _available_action_ids(dto: Mapping[str, Any]) -> set[str]:
    legal_actions = dto.get("legal_actions")
    if not isinstance(legal_actions, Sequence) or isinstance(legal_actions, (str, bytes)):
        return set()
    return {
        action_id
        for action in legal_actions
        if isinstance(action, Mapping) and action.get("is_available") is not False
        for action_id in [action.get("action_id")]
        if isinstance(action_id, str) and action_id
    }


def _action_type_for_id(dto: Mapping[str, Any], action_id: str) -> str | None:
    legal_actions = dto.get("legal_actions")
    if not isinstance(legal_actions, Sequence) or isinstance(legal_actions, (str, bytes)):
        return None
    for action in legal_actions:
        if not isinstance(action, Mapping) or action.get("is_available") is False:
            continue
        if action.get("action_id") != action_id:
            continue
        action_type = action.get("action_type")
        return action_type if isinstance(action_type, str) else None
    return None


def _validated_values(values: Sequence[Any], *, expected: int) -> list[float]:
    if len(values) != expected:
        raise RuntimeError("ValueModel.evaluate_batch must return exactly one value per dto")

    normalized: list[float] = []
    for raw_value in values:
        if isinstance(raw_value, (bool, str, bytes)):
            raise RuntimeError("ValueModel.evaluate_batch must return finite numeric values")
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("ValueModel.evaluate_batch must return finite numeric values") from exc
        if not math.isfinite(value):
            raise RuntimeError("ValueModel.evaluate_batch must return finite numeric values")
        normalized.append(value)
    return normalized


def _client_unusable(client: Any) -> bool:
    return getattr(client, "pending_retry", None) is not None or getattr(
        client, "session_invalid", False
    )


def _available_action_types(dto: Mapping[str, Any]) -> set[str] | None:
    legal_actions = dto.get("legal_actions")
    if not isinstance(legal_actions, Sequence) or isinstance(legal_actions, (str, bytes)):
        return None

    action_types: set[str] = set()
    for action in legal_actions:
        if not isinstance(action, Mapping):
            return None
        if action.get("is_available") is False:
            continue
        action_type = action.get("action_type")
        if not isinstance(action_type, str):
            return None
        action_types.add(action_type)
    return action_types


def _is_beam_searchable(
    dto: Mapping[str, Any], allowed_action_types: frozenset[str]
) -> bool:
    action_types = _available_action_types(dto)
    return action_types is not None and bool(action_types) and action_types <= allowed_action_types


def _is_continuation_decision(dto: Mapping[str, Any]) -> bool:
    action_types = _available_action_types(dto)
    return action_types is not None and bool(action_types) and action_types <= _CONTINUATION_ACTION_TYPES


def _is_terminal(dto: Mapping[str, Any]) -> bool:
    if dto.get("run_terminal") is True or dto.get("terminal") is True:
        return True
    boundary = dto.get("boundary")
    if isinstance(boundary, str) and boundary in ("terminal", "run_terminal"):
        return True
    transition = dto.get("transition")
    if isinstance(transition, Mapping) and transition.get("kind") == "combat_completed":
        return True
    legal_actions = dto.get("legal_actions")
    if isinstance(legal_actions, Sequence) and not isinstance(legal_actions, (str, bytes)):
        if not any(
            isinstance(action, Mapping) and action.get("is_available") is not False
            for action in legal_actions
        ):
            return True
    return False
