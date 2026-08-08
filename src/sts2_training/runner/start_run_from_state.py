"""Entry point 2: resume a Whole Run instance from a fully-specified run snapshot
(`RunSnapshot` - see `scenario.py`) and drive it to completion.

KNOWN GAP (as of this writing): STS2_RL's `API/instance_whole_run.py::WholeRunInstance`
does not yet consume a snapshot field from `instance_config` - it always calls
`WholeRunSession.start_run(seed, character_id, ascension)`, i.e. a FRESH run, even
though `WholeRunSession.load_state(snapshot_json)` already exists and could be wired to
it (see `RunSnapshot`'s docstring). Calling `start_run_from_state()` therefore always
raises `RunSnapshotRestoreNotSupportedError` rather than silently starting a fresh run
while claiming to resume a specific one - this module's plumbing (config, CLI) is ready
to use the moment that RL-side wiring lands; only the guard below needs removing.
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
from sts2_training.decision.search_modes import SEARCH_MODES
from sts2_training.runner.episode import EpisodeResult
from sts2_training.runner.scenario import RunSnapshot

__all__ = ["RunSnapshotRestoreNotSupportedError", "start_run_from_state"]


class RunSnapshotRestoreNotSupportedError(NotImplementedError):
    """Raised by `start_run_from_state()` until STS2_RL's `WholeRunInstance` wires
    `instance_config["snapshot_json"]` into `WholeRunSession.load_state()` - see this
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
    """`search_mode`/`beam_max_depth` are accepted (not just `engine`) for signature
    symmetry with the other two entry points, so callers/CLI scripts don't need to
    special-case this one - unused until the guard below is lifted.
    """
    raise RunSnapshotRestoreNotSupportedError(
        "STS2_RL's WholeRunInstance does not yet resume from RunSnapshot.snapshot_json "
        "(it always starts a fresh run) - see this module's docstring for the RL-side "
        "wiring this entry point is waiting on."
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--decision-timeout", type=float, default=30.0)
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument(
        "--search-mode",
        choices=sorted(SEARCH_MODES),
        default=None,
        help="beam search preset (see decision.search_modes); default: standard",
    )
    parser.add_argument(
        "--beam-depth",
        type=int,
        default=None,
        help="override just the beam search depth of --search-mode",
    )
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
    result = asyncio.run(_run(_parse_args(argv)))
    print(
        json.dumps(
            {
                "instance_id": result.instance_id,
                "decisions_made": result.decisions_made,
                "elapsed_s": result.elapsed_s,
                "decision_sources": result.decision_sources,
                "final_dto": result.final_dto,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
