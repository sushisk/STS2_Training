"""Entry point 2: start a normal, from-scratch Whole Run (`NewRunConfig` - see
`scenario.py`) and drive it to completion.

Programmatic use::

    result = await start_new_run(client, character_id="IRONCLAD", ascension=0)

CLI use::

    python -m sts2_training.runner.start_new_run \
        --host 127.0.0.1 --port 8765 --character-id IRONCLAD
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path
from typing import Any

from sts2_training.api import AsyncTrainingApiClient, TcpConnection
from sts2_training.decision import CombatDecisionEngine
from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.runner._cli import add_common_arguments, configure_logging, print_result
from sts2_training.runner.episode import EpisodeResult, start_and_run
from sts2_training.runner.scenario import NewRunConfig
from sts2_training.selection_log import JsonlSelectionLogger

__all__ = ["start_new_run"]


async def start_new_run(
    client: Any,
    *,
    character_id: str,
    ascension: int = 0,
    seed: int | None = None,
    rng: random.Random | None = None,
    start_timeout_s: float = 30.0,
    decision_timeout_s: float = 30.0,
    max_decisions: int | None = None,
    engine: CombatDecisionEngine | None = None,
    search_mode: str | BeamSearchConfig | None = None,
    beam_max_depth: int | None = None,
) -> EpisodeResult:
    """`seed=None` (default) picks a fresh random seed via `rng` (or module-level
    `random`) so repeated calls produce different runs; pass `seed` to pin one.
    `search_mode`/`beam_max_depth` select the beam config; pass `engine` instead for
    full manual control (mutually exclusive, see `episode.build_engine`).
    """
    if seed is None:
        random_source = rng if rng is not None else random
        seed = random_source.randint(1, 2**31 - 1)
    config = NewRunConfig(character_id=character_id, ascension=ascension, seed=seed)
    return await start_and_run(
        client,
        config.to_instance_config(),
        start_timeout_s=start_timeout_s,
        decision_timeout_s=decision_timeout_s,
        max_decisions=max_decisions,
        engine=engine,
        search_mode=search_mode,
        beam_max_depth=beam_max_depth,
    )


def _run_result_event(result: EpisodeResult) -> dict[str, Any]:
    """Return the run-level record that makes a selection log self-contained."""
    return {
        "event": "run_result",
        "instance_id": result.instance_id,
        "decisions_made": result.decisions_made,
        "elapsed_s": result.elapsed_s,
        "outcome": result.final_dto.get("outcome"),
        "final_dto": result.final_dto,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--character-id", required=True)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None, help="omit for a fresh random seed each run")
    parser.add_argument(
        "--selection-log",
        type=Path,
        default=None,
        help=(
            "write selection audit events plus one final run_result record to this "
            "JSONL file (overwrite if it exists)"
        ),
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> EpisodeResult:
    connection = TcpConnection(host=args.host, port=args.port, connect_timeout_s=args.connect_timeout)
    selection_logger = (
        JsonlSelectionLogger(args.selection_log, append=False) if args.selection_log is not None else None
    )
    try:
        async with AsyncTrainingApiClient(connection, selection_logger=selection_logger) as client:
            result = await start_new_run(
                client,
                character_id=args.character_id,
                ascension=args.ascension,
                seed=args.seed,
                decision_timeout_s=args.decision_timeout,
                max_decisions=args.max_decisions,
                search_mode=args.search_mode,
                beam_max_depth=args.beam_depth,
            )
        if selection_logger is not None:
            selection_logger(_run_result_event(result))
        return result
    finally:
        if selection_logger is not None:
            selection_logger.close()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging(args.log_level)
    print_result(asyncio.run(_run(args)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
