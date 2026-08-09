"""Top-level entry points for starting an instance and driving it to completion,
layered on `sts2_training.decision.CombatDecisionEngine`. See `how_to_use.md`.

Two entry points sharing one loop (`EpisodeRunner`):

- `start_combat_from_state`: Combat from a fully-specified board (`CombatScenario`).
- `start_new_run`: a normal, from-scratch Whole Run (`NewRunConfig`).

Both accept `search_mode`/`beam_max_depth` to pick the beam config (see
`decision.search_modes`) without hand-constructing a `BeamSearchConfig`.

The executable `start_*` modules are lazy exports. Importing them eagerly here would
preload the target before `python -m sts2_training.runner.start_*` executes it, causing
runpy's "found in sys.modules" RuntimeWarning and executing the module in a second
namespace.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from sts2_training.decision.search_modes import DEFAULT_SEARCH_MODE, SEARCH_MODES, resolve_search_mode
from sts2_training.runner.episode import (
    EpisodeLimitExceeded,
    EpisodeResult,
    EpisodeRunner,
    build_engine,
    start_and_run,
)
from sts2_training.runner.scenario import CombatScenario, EnemyScenario, NewRunConfig

_LAZY_EXPORTS = {
    "start_combat_from_state": (
        "sts2_training.runner.start_combat_from_state",
        "start_combat_from_state",
    ),
    "start_new_run": ("sts2_training.runner.start_new_run", "start_new_run"),
}

__all__ = [
    "CombatScenario",
    "DEFAULT_SEARCH_MODE",
    "EnemyScenario",
    "EpisodeLimitExceeded",
    "EpisodeResult",
    "EpisodeRunner",
    "NewRunConfig",
    "SEARCH_MODES",
    "build_engine",
    "resolve_search_mode",
    "start_and_run",
    "start_combat_from_state",
    "start_new_run",
]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
