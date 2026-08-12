"""Top-level entry points for starting and collecting Training instances.

Ordinary episode runners and training-only oracle collection share the same Combat
decision stack. Executable modules remain lazy exports so ``python -m`` does not preload
the target module and trigger runpy's duplicate-module warning.
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
    "SelfPlayRunResult": ("sts2_training.runner.self_play", "SelfPlayRunResult"),
    "run_self_play_batch": ("sts2_training.runner.self_play", "run_self_play_batch"),
    "OracleEpisodeResult": (
        "sts2_training.runner.oracle_collection",
        "OracleEpisodeResult",
    ),
    "OracleEpisodeRunner": (
        "sts2_training.runner.oracle_collection",
        "OracleEpisodeRunner",
    ),
}

__all__ = [
    "CombatScenario",
    "DEFAULT_SEARCH_MODE",
    "EnemyScenario",
    "EpisodeLimitExceeded",
    "EpisodeResult",
    "EpisodeRunner",
    "NewRunConfig",
    "OracleEpisodeResult",
    "OracleEpisodeRunner",
    "SEARCH_MODES",
    "SelfPlayRunResult",
    "build_engine",
    "resolve_search_mode",
    "run_self_play_batch",
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
