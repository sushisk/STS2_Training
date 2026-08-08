"""Entry point 2: resume a Whole Run instance from a fully-specified run snapshot
(`RunSnapshot` - see `scenario.py`) and drive it to completion.

KNOWN GAP: STS2_RL's `WholeRunInstance` doesn't consume a snapshot field yet - it
always starts a fresh run (see `RunSnapshot`'s docstring). `start_run_from_state()`
always raises `RunSnapshotRestoreNotSupportedError` rather than silently starting
fresh under a "resumed" label; the plumbing here is otherwise ready to use once that
RL-side wiring lands.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from sts2_training.api import AsyncTrainingApiClient, TcpConnection
from sts2_training.decision import CombatDecisionEngine
from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.runner._cli import add_common_arguments, print_result
from sts2_training.runner.episode import EpisodeResult
from sts2_training.runner.scenario import RunSnapshot

__all__ = ["RunSnapshotRestoreNotSupportedError", "start_run_from_state"]


class RunSnapshotRestoreNotSupportedError(NotImplementedError):
    """Raised until STS2_RL's `WholeRunInstance` wires up snapshot restore - see this
    module's docstring."""


async def start_run_from_state(
    client: Any,
    snapshot: RunSnapshot,
    *,
    start_timeout_s: float = 30.0,
    decision_timeout_s: float = 30.0,
    max_decisions: int | None = None,
    engine: CombatDecisionEngine | None = None,
    search_mode: str | BeamSearchConfig | None = None,
    beam_max_depth: int | None = None,
) -> EpisodeResult:
    """`search_mode`/`beam_max_depth`/`engine` are accepted for signature symmetry
    with the other two entry points - unused until the guard below is lifted.
    """
    raise RunSnapshotRestoreNotSupportedError(
        "STS2_RL's WholeRunInstance does not yet resume from RunSnapshot.snapshot_json "
        "(it always starts a fresh run) - see this module's docstring for the RL-side "
        "wiring this entry point is waiting on."
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="path to a JSON file with character_id, ascension, seed, and "
        "snapshot_json (WholeRunSession.save_state() output)",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> EpisodeResult:
    data = json.loads(args.snapshot.read_text(encoding="utf-8"))
    snapshot = RunSnapshot(**data)
    connection = TcpConnection(host=args.host, port=args.port, connect_timeout_s=args.connect_timeout)
    async with AsyncTrainingApiClient(connection) as client:
        return await start_run_from_state(
            client,
            snapshot,
            decision_timeout_s=args.decision_timeout,
            max_decisions=args.max_decisions,
            search_mode=args.search_mode,
            beam_max_depth=args.beam_depth,
        )


def main(argv: list[str] | None = None) -> int:
    print_result(asyncio.run(_run(_parse_args(argv))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
