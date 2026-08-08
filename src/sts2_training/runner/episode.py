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
import time
from dataclasses import dataclass, field
from typing import Any

from sts2_training.decision import CombatDecisionEngine
from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.decision.search_modes import resolve_search_mode

JsonObject = dict[str, Any]

_LOG = logging.getLogger(__name__)

__all__ = ["EpisodeLimitExceeded", "EpisodeResult", "EpisodeRunner", "build_engine"]


def build_engine(
    client: Any,
    *,
    engine: CombatDecisionEngine | None = None,
    search_mode: str | BeamSearchConfig | None = None,
    beam_max_depth: int | None = None,
) -> CombatDecisionEngine:
    """`engine` given -> used as-is; combining it with `search_mode`/`beam_max_depth`
    raises rather than silently ignoring one. Otherwise builds a fresh
    `CombatDecisionEngine` from `search_mode`/`beam_max_depth` (see
    `decision.search_modes.resolve_search_mode`).
    """
    if engine is not None:
        if search_mode is not None or beam_max_depth is not None:
            raise ValueError(
                "pass either `engine` or `search_mode`/`beam_max_depth`, not both - "
                "an explicit engine already has its own BeamSearchConfig"
            )
        return engine
    beam_config = resolve_search_mode(search_mode, max_depth=beam_max_depth)
    return CombatDecisionEngine(client, beam_config=beam_config)


class EpisodeLimitExceeded(RuntimeError):
    """Raised when an episode exceeds `max_decisions` without reaching a terminal
    state - a runaway-loop guard. `None` (default) means no limit."""


@dataclass(frozen=True)
class EpisodeResult:
    """What actually happened when an instance was driven to completion.

    `final_dto` is the raw `masked_emulator_dto` from the last response seen, even if
    that's the very first one (an instance already terminal at the start of `run()`).
    Not collapsed into a guessed `"victory"/"defeat"` enum - read `final_dto["outcome"]`
    for that.
    """

    instance_id: str
    decisions_made: int
    final_dto: JsonObject
    elapsed_s: float
    # Count of DecisionOutcome.source values seen - a cheap signal for how often beam
    # search actually won out versus falling back.
    decision_sources: dict[str, int] = field(default_factory=dict)


class EpisodeRunner:
    """Unlike `CombatDecisionEngine`/`BeamSearchEngine`, holds no instance-scoped
    state (no branch-id allocator to keep unique), so one instance is fine to reuse
    across multiple sequential episodes on the same client.
    """

    def __init__(self, client: Any, engine: CombatDecisionEngine | None = None) -> None:
        self._client = client
        self._engine = engine or CombatDecisionEngine(client)

    async def run(
        self,
        instance_id: str,
        *,
        decision_timeout_s: float,
        max_decisions: int | None = None,
        close_timeout_s: float = 10.0,
    ) -> EpisodeResult:
        """Repeatedly decide+commit until `legal_actions` is empty (Combat
        victory/defeat, or Whole Run `run_terminal`), then best-effort
        `close_instance` regardless of how the loop ended. Inlines
        `decide()`+`commit_action()` (what `decide_and_commit()` does) just to
        capture `DecisionOutcome.source` for `decision_sources` along the way.
        """
        t_start = time.monotonic()
        decisions_made = 0
        decision_sources: dict[str, int] = {}
        final_dto: JsonObject = {}

        try:
            while True:
                outcome = await self._engine.decide(instance_id, timeout_s=decision_timeout_s)
                decision_sources[outcome.source] = decision_sources.get(outcome.source, 0) + 1
                if outcome.chosen_action_id is None:
                    # Already terminal before this decide() - nothing to commit, but
                    # still report the dto decide() fetched rather than leaving
                    # final_dto at its {} default.
                    final_dto = outcome.decision.get("masked_emulator_dto") or {}
                    break

                response = await self._client.commit_action(
                    instance_id,
                    outcome.decision["decision_point_id"],
                    outcome.chosen_action_id,
                    timeout_s=decision_timeout_s,
                )
                decisions_made += 1
                final_dto = response.get("masked_emulator_dto") or {}

                if not final_dto.get("legal_actions"):
                    break
                if max_decisions is not None and decisions_made >= max_decisions:
                    raise EpisodeLimitExceeded(
                        f"instance_id={instance_id} exceeded max_decisions={max_decisions} "
                        "without reaching a terminal state"
                    )
        finally:
            await self._close_best_effort(instance_id, close_timeout_s)

        return EpisodeResult(
            instance_id=instance_id,
            decisions_made=decisions_made,
            final_dto=final_dto,
            elapsed_s=time.monotonic() - t_start,
            decision_sources=decision_sources,
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
