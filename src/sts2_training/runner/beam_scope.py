"""The single place a runner widens a budget preset to the Combat Beam semantic scope.

`resolve_search_mode()` returns *budget-shaped* `BeamSearchConfig` objects whose
`beam_searchable_action_types` stays at the conservative dataclass default
(`{"system", "card", "potion"}`) - it deliberately does not encode decision-phase
semantics. `CombatDecisionEngine` treats an explicitly supplied `beam_config` as
authoritative, so a runner that passes a resolved preset straight through silently keeps
that narrow scope and `BeamSearchEngine` then drops every branch whose resulting state is
an interactive continuation.

That is not a cosmetic difference. The Emulator publishes a `choice_target` continuation
for `TargetType.AnyEnemy` cards exactly when two or more enemies are alive, so a narrow
scope removes every targeted attack from the search in multi-enemy fights while leaving
self-targeted cards untouched - and it does so without producing a fault.

Every runner that builds an engine from a named/default search mode must route its config
through `runner_combat_beam_config` so the scope cannot drift per entry point.
"""

from __future__ import annotations

from dataclasses import replace

from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.decision.combat_decision import COMBAT_BEAM_ACTION_TYPES

__all__ = ["runner_combat_beam_config"]


def runner_combat_beam_config(config: BeamSearchConfig) -> BeamSearchConfig:
    """Return ``config`` with the runner's full Combat semantic scope applied.

    Only the semantic scope is changed; every budget knob (depth, width, top-k, time
    budget) is preserved. `simulation_options` is copied so callers cannot share mutable
    state between engines. Whether Beam actually executes stays a separate decision
    (`search_mode_uses_beam`).
    """

    return replace(
        config,
        beam_searchable_action_types=COMBAT_BEAM_ACTION_TYPES,
        simulation_options=(
            None if config.simulation_options is None else dict(config.simulation_options)
        ),
    )
