"""`BeamSearchEngine`: policy-guided beam search over `AsyncTrainingApiClient`.

Per `docs/STS2_wire_contract_v0.7.md`'s Beam integration guidance, **one beam
depth is sent as one or more bounded `emulate_actions` requests**: every
surviving beam node's policy-proposed candidate actions are batched together
and scored together, rather than one `emulate_action` call per candidate. A
single request cannot exceed RL's `BranchManager.max_branches` capacity (64
for the standard Combat configuration - see `BeamSearchConfig.max_batch_size`),
so a frontier wider than that is chunked into multiple same-depth requests.
This batching is what makes the assumed latency model (`beam search ~= 5ms +
1ms/decision`, amortized over every board in the batch) achievable.

`emulate_actions` is synchronous at the RL coordinator boundary -
`BranchManager.poll()` only returns once every Branch dispatched by the call
has reached a terminal outcome, so a response's `branch_results` entries are
always `completed`/`partial`/`faulted`, never `queued`/`running`. This engine
does not poll for stragglers; there are none to poll for.

Branches created during a search are Training-owned bookkeeping the RL side
needs cleaned up explicitly (`cancel_branches` + `release_branches`) once
they're no longer needed - this engine always does that in a `finally` block,
best-effort, even on error or timeout.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from sts2_training.api.contract import ApiOperationError, ApiProtocolError, ROOT_BRANCH_ID
from sts2_training.decision.policy import ActionCandidate, PolicyModel
from sts2_training.decision.value import ValueModel

JsonObject = dict[str, Any]

_LOG = logging.getLogger(__name__)

_RESOLVED_STATUSES = ("completed", "partial")


@dataclass
class BeamSearchConfig:
    beam_width: int = 8
    top_k_actions: int = 4
    max_depth: int = 2
    simulation_options: Mapping[str, Any] | None = None
    time_budget_ms: float | None = None
    # RL's standard Combat `BranchManager.max_branches` capacity (see
    # `docs/STS2_wire_contract_v0.7.md`'s "Batch-size capability" section). A
    # frontier wider than this is split into multiple same-depth
    # `emulate_actions` requests rather than sent as one oversized request.
    max_batch_size: int = 64
    expand_partial: bool = True
    release_branches_on_finish: bool = True
    beam_searchable_action_types: frozenset[str] = field(
        default_factory=lambda: frozenset({"system", "card", "potion"})
    )

    def __post_init__(self) -> None:
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if self.top_k_actions <= 0:
            raise ValueError("top_k_actions must be positive")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
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
    """Issues branch_id/rng_id values unique for as long as this allocator
    lives.

    Branch IDs are "instance内で生涯一意" per the RL/Training DTO contract -
    never reused, even after cancel/release. Construct ONE allocator per API
    client/instance (`BeamSearchEngine` owns one for its own lifetime) and
    reuse it across every real decision's beam search, not just within a
    single `search()` call.
    """

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
    """Construct once per `AsyncTrainingApiClient` instance/session and reuse
    across every real decision - see `BranchIdAllocator`.
    """

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
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        cfg = self.config
        deadline = time.monotonic() + timeout_s
        if cfg.time_budget_ms is not None:
            deadline = min(deadline, time.monotonic() + cfg.time_budget_ms / 1000.0)

        stats = BeamSearchStats()
        t_start = time.monotonic()

        root_dto = root_decision.get("masked_emulator_dto")
        root_decision_point_id = root_decision.get("decision_point_id")
        if not isinstance(root_dto, Mapping) or not isinstance(root_decision_point_id, str):
            return BeamSearchResult(None, None, None, "invalid_root_decision", stats)

        root_legal_actions = root_dto.get("legal_actions")
        if not root_legal_actions:
            return BeamSearchResult(None, None, None, "no_legal_actions", stats)

        action_types = {a.get("action_type") for a in root_legal_actions}
        if not action_types <= cfg.beam_searchable_action_types:
            return BeamSearchResult(None, None, None, "not_beam_searchable", stats)

        root_node = BeamNode(
            branch_id=ROOT_BRANCH_ID,
            parent_branch_id=ROOT_BRANCH_ID,
            rng_id=0,
            decision_point_id=root_decision_point_id,
            masked_emulator_dto=root_dto,
            depth=0,
            value=self._value_fn.evaluate(root_dto),
            root_action_id=None,
        )

        beam: list[BeamNode] = [root_node]
        finished: list[BeamNode] = []
        all_branch_ids: list[str] = []
        reason = "max_depth"

        try:
            for depth in range(cfg.max_depth):
                if not beam:
                    reason = "beam_exhausted"
                    break
                if time.monotonic() >= deadline:
                    reason = "time_budget"
                    break

                items, item_meta, policy_ms = self._propose_frontier(beam)
                stats.policy_ms += policy_ms
                if not items:
                    reason = "no_candidates"
                    break

                branch_results, fatal_reason = await self._emulate_depth_batch(
                    instance_id, items, item_meta, all_branch_ids, stats, deadline
                )

                next_beam, newly_finished, value_ms = self._score_frontier(
                    item_meta, branch_results, depth
                )
                stats.value_ms += value_ms
                stats.nodes_expanded += len(item_meta)
                finished.extend(newly_finished)

                next_beam.sort(key=lambda n: n.value, reverse=True)
                beam = next_beam[: cfg.beam_width]
                stats.depths_completed += 1

                if fatal_reason is not None:
                    # A chunk of this depth's batch failed (rejected, or ran out of
                    # time budget mid-depth). Keep whatever chunks DID resolve (already
                    # folded into finished/beam above) and stop going deeper.
                    reason = fatal_reason
                    break

            finished.extend(beam)
        finally:
            t0 = time.monotonic()
            await self._cleanup(instance_id, all_branch_ids)
            stats.cleanup_ms += (time.monotonic() - t0) * 1000.0

        stats.total_ms = (time.monotonic() - t_start) * 1000.0

        if not finished:
            return BeamSearchResult(None, None, None, reason, stats)
        best_node = max(finished, key=lambda n: n.value)
        return BeamSearchResult(best_node.root_action_id, best_node.value, best_node, reason, stats)

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
            for candidate in candidates:
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
        """Sends one depth's `items` to RL as one or more `emulate_actions`
        chunks of at most `max_batch_size` each (see the v0.7 wire contract's
        `max_branches` capacity note), merging `branch_results` across chunks.

        Returns `(merged_branch_results, fatal_reason)`. `fatal_reason` is
        `None` on full success; otherwise it names why chunking stopped early,
        and `merged_branch_results` still holds whatever chunks succeeded
        before that - callers should score those, not discard them.
        """
        cfg = self.config
        branch_results: dict[str, Any] = {}
        for start in range(0, len(items), cfg.max_batch_size):
            if time.monotonic() >= deadline:
                return branch_results, "time_budget"
            chunk_items = items[start : start + cfg.max_batch_size]
            chunk_meta = item_meta[start : start + cfg.max_batch_size]

            t0 = time.monotonic()
            try:
                response = await self._client.emulate_actions(
                    instance_id,
                    chunk_items,
                    timeout_s=_remaining(deadline),
                    simulation_options=cfg.simulation_options,
                )
            except (ApiOperationError, ApiProtocolError) as exc:
                # Whole-chunk admission failure: per the RL/Training contract, no
                # Branch was created for ANY item in this chunk. Do NOT add these
                # branch_ids to cleanup (they don't exist on RL); prior chunks in
                # this same depth already did succeed, so keep their results.
                stats.emulate_actions_ms += (time.monotonic() - t0) * 1000.0
                return branch_results, f"emulate_actions_rejected:{type(exc).__name__}"
            stats.emulate_actions_ms += (time.monotonic() - t0) * 1000.0

            all_branch_ids.extend(meta[2] for meta in chunk_meta)
            stats.branches_created += len(chunk_meta)
            branch_results.update(response.get("branch_results") or {})
        return branch_results, None

    def _score_frontier(
        self,
        item_meta: Sequence[tuple[BeamNode, ActionCandidate, str, int]],
        branch_results: Mapping[str, Any],
        depth: int,
    ) -> tuple[list[BeamNode], list[BeamNode], float]:
        cfg = self.config
        resolved: list[tuple[BeamNode, ActionCandidate, str, int, Mapping[str, Any], Mapping[str, Any], Any]] = []
        for node, candidate, branch_id, rng_id in item_meta:
            result = branch_results.get(branch_id)
            if not isinstance(result, Mapping) or result.get("status") not in _RESOLVED_STATUSES:
                continue
            dto = result.get("masked_emulator_dto")
            if not isinstance(dto, Mapping):
                continue
            resolved.append((node, candidate, branch_id, rng_id, result, dto, result.get("status")))

        t0 = time.monotonic()
        values = self._value_fn.evaluate_batch([entry[5] for entry in resolved]) if resolved else []
        value_ms = (time.monotonic() - t0) * 1000.0

        next_beam: list[BeamNode] = []
        newly_finished: list[BeamNode] = []
        is_last_depth = depth + 1 >= cfg.max_depth
        for (node, candidate, branch_id, rng_id, result, dto, status), value in zip(resolved, values):
            decision_point_id = result.get("decision_point_id")
            new_node = BeamNode(
                branch_id=branch_id,
                parent_branch_id=node.branch_id,
                rng_id=rng_id,
                decision_point_id=decision_point_id if isinstance(decision_point_id, str) else "",
                masked_emulator_dto=dto,
                depth=depth + 1,
                value=value,
                root_action_id=node.root_action_id or candidate.action_id,
                branch_log=tuple(result.get("branch_log") or ()),
                terminal=_is_terminal(dto),
            )
            cannot_expand = not new_node.decision_point_id or (
                status == "partial" and not cfg.expand_partial
            )
            if new_node.terminal or is_last_depth or cannot_expand:
                newly_finished.append(new_node)
            else:
                next_beam.append(new_node)
        return next_beam, newly_finished, value_ms

    async def _cleanup(self, instance_id: str, branch_ids: list[str]) -> None:
        if not branch_ids or not self.config.release_branches_on_finish:
            return
        if getattr(self._client, "pending_retry", None) is not None or getattr(
            self._client, "session_invalid", False
        ):
            _LOG.warning(
                "skipping beam search branch cleanup for instance_id=%s: client has "
                "pending_retry or an invalid session",
                instance_id,
            )
            return
        try:
            await self._client.cancel_branches(instance_id, branch_ids, timeout_s=5.0)
            await self._client.release_branches(instance_id, branch_ids, timeout_s=5.0)
        except Exception:  # noqa: BLE001 - best-effort cleanup must never mask the search result
            _LOG.exception(
                "beam search branch cleanup failed for instance_id=%s (%d branches)",
                instance_id,
                len(branch_ids),
            )


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
    if legal_actions is not None and len(legal_actions) == 0:
        return True
    return False


def _remaining(deadline: float, *, minimum: float = 0.05) -> float:
    return max(minimum, deadline - time.monotonic())
