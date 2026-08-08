"""`EpisodeRunner`: the one "drive an already-started instance to completion" loop
shared by every entry point in this package (`start_combat_from_state`,
`start_run_from_state`, `start_new_run`).

The three entry points differ only in HOW an instance gets created (what
`instance_config` they build - see `scenario.py`); once `start_instance` has
returned an `instance_id`, driving it to completion is identical regardless of
`instance_type`. `CombatDecisionEngine` (see `sts2_training.decision`) already
generalizes over this: Beam Search only ever activates for `card`/`potion`/`system`
decisions, and everything else (including Whole Run's `map_select`/`event_choice`/
`shop_choice`/`rest_choice`/`reward_select` boundaries) falls through to
`HeuristicCombatSelector`, which classifies by `action_type` generically rather than
assuming Combat. So this module adds no new decision logic - only the loop/lifecycle
around it.
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
    """Resolves the engine each of the three top-level entry points should use.

    - `engine` given: used as-is. Combining it with `search_mode`/`beam_max_depth` is
      rejected (`ValueError`) rather than silently ignoring one of them - an explicit
      engine already carries its own `BeamSearchConfig`.
    - otherwise: a fresh `CombatDecisionEngine` built from `search_mode`/
      `beam_max_depth` (see `decision.search_modes.resolve_search_mode` for how those
      two combine - `beam_max_depth` overrides just the depth of whichever mode was
      chosen).
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
    state - a runaway-loop guard, not a normal outcome. `None` (the default) means
    no limit."""


@dataclass(frozen=True)
class EpisodeResult:
    """What actually happened when an instance was driven to completion.

    `final_dto` is the raw `masked_emulator_dto` from the last response seen (empty
    `legal_actions`, or a Whole Run `run_terminal` payload), even if that is the very
    first response observed (an instance already terminal at the start of `run()`).
    This module deliberately does not collapse it into a guessed `"victory"/"defeat"`
    enum: `outcome` is present on both boundaries as of STS2_RL's
    `agent/expose-terminal-outcome`, but reading it here would silently re-couple this
    module to that wire detail - callers that need a definitive win/loss read
    `final_dto["outcome"]` themselves.
    """

    instance_id: str
    decisions_made: int
    final_dto: JsonObject
    elapsed_s: float
    # Count of DecisionOutcome.source values seen ("beam_search"/"heuristic_fallback"/
    # "forced_single_action"/"none") - a cheap health signal for how often beam search
    # actually won out versus falling back, without needing to instrument the caller.
    decision_sources: dict[str, int] = field(default_factory=dict)


class EpisodeRunner:
    """Construct one per `(client, engine)` pair and call `run()` once per instance -
    unlike `CombatDecisionEngine`/`BeamSearchEngine`, this class holds no
    instance-scoped state itself (no branch-id allocator to keep unique), so reuse
    across multiple sequential instances on the same client is fine.
    """

    def __init__(self, client: Any, engine: CombatDecisionEngine | None = None) -> None:
        self._client = client
        self._engine = engine or CombatDecisionEngine(client)

    @property
    def engine(self) -> CombatDecisionEngine:
        return self._engine

    async def run(
        self,
        instance_id: str,
        *,
        decision_timeout_s: float,
        max_decisions: int | None = None,
        close_timeout_s: float = 10.0,
    ) -> EpisodeResult:
        """Repeatedly decide+commit until the instance reaches a state with no
        `legal_actions` (Combat victory/defeat, or Whole Run `run_terminal`), then
        best-effort `close_instance`s regardless of how the loop ended. Inlines
        `CombatDecisionEngine.decide()` + `client.commit_action()` (equivalent to
        `decide_and_commit()`) rather than calling that convenience method directly,
        purely to capture `DecisionOutcome.source` for `EpisodeResult.decision_sources`
        along the way.
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
                    # No legal_actions at all - instance was already terminal before
                    # this iteration's decide() even ran, so there is nothing to
                    # commit. Still capture the dto `decide()` already fetched (rather
                    # than leaving `final_dto` at its {} default) so a scenario that
                    # starts pre-terminal is reported accurately instead of empty.
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
