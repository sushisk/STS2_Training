"""`EpisodeRunner`: the "drive an already-started instance to completion" loop
shared by every entry point in this package.

The three entry points differ only in how an instance gets created (see
`scenario.py`); once `start_instance` returns an `instance_id`, driving it to
completion is identical regardless of `instance_type` - `CombatDecisionEngine`
already generalizes over that (Beam Search only for `card`/`potion`/`system`,
everything else falls through to `HeuristicCombatSelector`), so this module adds
no new decision logic, only the loop/lifecycle around it.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sts2_training.decision import CombatDecisionEngine
from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.decision.search_modes import resolve_search_mode
from sts2_training.selection.heuristic_selector import NoAvailableActionError

JsonObject = dict[str, Any]

_LOG = logging.getLogger(__name__)

__all__ = ["EpisodeLimitExceeded", "EpisodeResult", "EpisodeRunner", "build_engine", "start_and_run"]


def _validate_max_decisions(max_decisions: int | None) -> None:
    if max_decisions is not None and (
        isinstance(max_decisions, bool)
        or not isinstance(max_decisions, int)
        or max_decisions <= 0
    ):
        raise ValueError("max_decisions must be a positive integer or None")


def _validate_timeout(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")


def _is_terminal_dto(dto: Mapping[str, Any]) -> bool:
    return dto.get("terminal") is True or dto.get("run_terminal") is True


def _terminal_dto_from_no_action(error: NoAvailableActionError) -> JsonObject | None:
    decision = error.decision
    if not isinstance(decision, Mapping):
        return None
    dto = decision.get("masked_emulator_dto")
    if not isinstance(dto, Mapping) or not _is_terminal_dto(dto):
        return None
    return dict(dto)


def _validate_engine_client(client: Any, engine: CombatDecisionEngine) -> None:
    if engine.client is not client:
        raise ValueError(
            "explicit CombatDecisionEngine must be bound to the same client passed to the runner"
        )


def build_engine(
    client: Any,
    *,
    engine: CombatDecisionEngine | None = None,
    search_mode: str | BeamSearchConfig | None = None,
    beam_max_depth: int | None = None,
) -> CombatDecisionEngine:
    """`engine` given -> used as-is after verifying it belongs to `client`; combining
    it with `search_mode`/`beam_max_depth` raises rather than silently ignoring one.
    Otherwise builds a fresh `CombatDecisionEngine` from `search_mode`/
    `beam_max_depth` (see `decision.search_modes.resolve_search_mode`).
    """
    if engine is not None:
        if search_mode is not None or beam_max_depth is not None:
            raise ValueError(
                "pass either `engine` or `search_mode`/`beam_max_depth`, not both - "
                "an explicit engine already has its own BeamSearchConfig"
            )
        _validate_engine_client(client, engine)
        return engine
    beam_config = resolve_search_mode(search_mode, max_depth=beam_max_depth)
    return CombatDecisionEngine(client, beam_config=beam_config)


class EpisodeLimitExceeded(RuntimeError):
    """Raised when an episode reaches `max_decisions` without reaching a terminal
    state - a runaway-loop guard. `None` (default) means no limit."""


@dataclass(frozen=True)
class EpisodeResult:
    """What actually happened when an instance was driven to completion.

    `final_dto` is the raw terminal `masked_emulator_dto` last observed - not
    collapsed into a guessed `"victory"/"defeat"` enum; read `final_dto["outcome"]`
    for that. This includes an instance that was already terminal before the first
    commit.
    """

    instance_id: str
    decisions_made: int
    final_dto: JsonObject
    elapsed_s: float


class EpisodeRunner:
    """Unlike `CombatDecisionEngine`/`BeamSearchEngine`, holds no instance-scoped
    state (no branch-id allocator to keep unique), so one instance is fine to reuse
    across multiple sequential episodes on the same client.
    """

    def __init__(self, client: Any, engine: CombatDecisionEngine | None = None) -> None:
        self._client = client
        if engine is not None:
            _validate_engine_client(client, engine)
            self._engine = engine
        else:
            self._engine = CombatDecisionEngine(client)

    async def run(
        self,
        instance_id: str,
        *,
        decision_timeout_s: float,
        max_decisions: int | None = None,
        close_timeout_s: float = 10.0,
    ) -> EpisodeResult:
        """Repeatedly `decide_and_commit` until an explicit Combat/Whole Run terminal
        marker is observed, then best-effort `close_instance` regardless of how the
        loop ended. A non-terminal decision with no selectable action fails closed.
        """
        _validate_max_decisions(max_decisions)
        _validate_timeout("decision_timeout_s", decision_timeout_s)
        _validate_timeout("close_timeout_s", close_timeout_s)
        t_start = time.monotonic()
        decisions_made = 0
        final_dto: JsonObject = {}

        try:
            while True:
                try:
                    response = await self._engine.decide_and_commit(
                        instance_id, timeout_s=decision_timeout_s
                    )
                except NoAvailableActionError as exc:
                    terminal_dto = _terminal_dto_from_no_action(exc)
                    if terminal_dto is None:
                        raise
                    final_dto = terminal_dto
                    break

                decisions_made += 1
                if not isinstance(response, Mapping):
                    raise RuntimeError("commit_action must return a mapping")
                raw_final_dto = response.get("masked_emulator_dto")
                if not isinstance(raw_final_dto, Mapping):
                    raise RuntimeError("commit_action returned an invalid masked_emulator_dto")
                final_dto = dict(raw_final_dto)

                if _is_terminal_dto(final_dto):
                    break
                if not final_dto.get("legal_actions"):
                    raise NoAvailableActionError(
                        "non-terminal decision has no legal_actions to select from",
                        decision=response,
                    )
                if max_decisions is not None and decisions_made >= max_decisions:
                    raise EpisodeLimitExceeded(
                        f"instance_id={instance_id} reached max_decisions={max_decisions} "
                        "without reaching a terminal state"
                    )
        finally:
            await self._close_best_effort(instance_id, close_timeout_s)

        return EpisodeResult(
            instance_id=instance_id,
            decisions_made=decisions_made,
            final_dto=final_dto,
            elapsed_s=time.monotonic() - t_start,
        )

    async def _close_best_effort(self, instance_id: str, timeout_s: float) -> None:
        if getattr(self._client, "pending_retry", None) is not None or getattr(
            self._client, "session_invalid", False
        ):
            _LOG.warning(
                "skipping close_instance for instance_id=%s: client has pending_retry "
                "or an invalid session",
                instance_id,
            )
            return
        try:
            await self._client.close_instance(instance_id, timeout_s=timeout_s)
        except Exception:  # noqa: BLE001 - best-effort cleanup must never mask the episode result
            _LOG.exception("close_instance failed for instance_id=%s", instance_id)


async def start_and_run(
    client: Any,
    instance_config: JsonObject,
    *,
    start_timeout_s: float = 30.0,
    decision_timeout_s: float = 30.0,
    max_decisions: int | None = None,
    engine: CombatDecisionEngine | None = None,
    search_mode: str | BeamSearchConfig | None = None,
    beam_max_depth: int | None = None,
) -> EpisodeResult:
    """`build_engine` + `start_instance` + `EpisodeRunner.run` - the shared body
    behind every `start_*` entry point in this package; each just supplies its own
    `instance_config` (see `scenario.py`).
    """
    _validate_max_decisions(max_decisions)
    _validate_timeout("start_timeout_s", start_timeout_s)
    _validate_timeout("decision_timeout_s", decision_timeout_s)
    resolved_engine = build_engine(
        client, engine=engine, search_mode=search_mode, beam_max_depth=beam_max_depth
    )
    instance_id = await client.start_instance(instance_config, timeout_s=start_timeout_s)
    return await EpisodeRunner(client, resolved_engine).run(
        instance_id, decision_timeout_s=decision_timeout_s, max_decisions=max_decisions
    )
