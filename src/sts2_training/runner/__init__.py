"""Top-level entry points for starting an instance and driving it to completion,
layered on `sts2_training.decision.CombatDecisionEngine`. See `how_to_use.md`.

Three entry points, one management role each, sharing one loop (`EpisodeRunner`):

- `start_combat_from_state`: Combat from a fully-specified board (`CombatScenario`).
- `start_run_from_state`: Whole Run resumed from a fully-specified snapshot
  (`RunSnapshot`) - currently always raises `RunSnapshotRestoreNotSupportedError`,
  see that module's docstring for the pending RL-side dependency.
- `start_new_run`: a normal, from-scratch Whole Run (`NewRunConfig`).
"""

from sts2_training.runner.episode import EpisodeLimitExceeded, EpisodeResult, EpisodeRunner
from sts2_training.runner.scenario import CombatScenario, EnemyScenario, NewRunConfig, RunSnapshot
from sts2_training.runner.start_combat_from_state import start_combat_from_state
from sts2_training.runner.start_new_run import start_new_run
from sts2_training.runner.start_run_from_state import (
    RunSnapshotRestoreNotSupportedError,
    start_run_from_state,
)

__all__ = [
    "CombatScenario",
    "EnemyScenario",
    "EpisodeLimitExceeded",
    "EpisodeResult",
    "EpisodeRunner",
    "NewRunConfig",
    "RunSnapshot",
    "RunSnapshotRestoreNotSupportedError",
    "start_combat_from_state",
    "start_new_run",
    "start_run_from_state",
]
