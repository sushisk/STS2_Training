"""Entry point 3: start a normal, from-scratch Whole Run (`NewRunConfig` - see
`scenario.py`) and drive it to completion.

Programmatic use::

    result = await start_new_run(client, character_id="IRONCLAD", ascension=0)

CLI use::

    python -m sts2_training.runner.start_new_run \\
        --host 127.0.0.1 --port 8765 --character-id IRONCLAD
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from typing import Any

from sts2_training.api import AsyncTrainingApiClient, TcpConnection
from sts2_training.decision import CombatDecisionEngine
from sts2_training.runner.episode import EpisodeResult, EpisodeRunner
from sts2_training.runner.scenario import NewRunConfig

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
) -> EpisodeResult:
    """`seed=None` (the default) picks a fresh random seed - via `rng` if given,
    otherwise the module-level `random` - so repeated calls produce different runs.
    Pass an explicit `seed` to pin a specific, reproducible run instead.
    """
    if seed is None:
        seed = (rng or random).randint(1, 2**31 - 1)
    config = NewRunConfig(character_id=character_id, ascension=ascension, seed=seed)
    instance_id = await client.start_instance(
        config.to_instance_config(), timeout_s=start_timeout_s
    )
    runner = EpisodeRunner(client, engine)
    return await runner.run(
        instance_id, decision_timeout_s=decision_timeout_s, max_decisions=max_decisions
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--decision-timeout", type=float, default=30.0)
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument("--character-id", required=True)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None, help="omit for a fresh random seed each run")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> EpisodeResult:
    connection = TcpConnection(host=args.host, port=args.port, connect_timeout_s=args.connect_timeout)
    async with AsyncTrainingApiClient(connection) as client:
        return await start_new_run(
            client,
            character_id=args.character_id,
            ascension=args.ascension,
            seed=args.seed,
            decision_timeout_s=args.decision_timeout,
            max_decisions=args.max_decisions,
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
