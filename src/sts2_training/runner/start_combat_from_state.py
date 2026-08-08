"""Entry point 1: start a Combat instance from a fully-specified board state
(`CombatScenario` - see `scenario.py`) and drive it to completion.

Programmatic use::

    result = await start_combat_from_state(client, scenario, decision_timeout_s=30.0)

CLI use::

    python -m sts2_training.runner.start_combat_from_state \\
        --host 127.0.0.1 --port 8765 --scenario path/to/scenario.json
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
from sts2_training.runner.episode import EpisodeResult, EpisodeRunner, build_engine
from sts2_training.runner.scenario import CombatScenario, EnemyScenario

__all__ = ["start_combat_from_state"]


async def start_combat_from_state(
    client: Any,
    scenario: CombatScenario,
    *,
    start_timeout_s: float = 30.0,
    decision_timeout_s: float = 30.0,
    max_decisions: int | None = None,
    engine: CombatDecisionEngine | None = None,
    search_mode: str | BeamSearchConfig | None = None,
    beam_max_depth: int | None = None,
) -> EpisodeResult:
    """`client` must not already have an active instance. `search_mode`/
    `beam_max_depth` select the beam config (see `decision.search_modes`); pass
    `engine` instead for full manual control (mutually exclusive, see `build_engine`).
    """
    resolved_engine = build_engine(
        client, engine=engine, search_mode=search_mode, beam_max_depth=beam_max_depth
    )
    instance_id = await client.start_instance(
        scenario.to_instance_config(), timeout_s=start_timeout_s
    )
    runner = EpisodeRunner(client, resolved_engine)
    return await runner.run(
        instance_id, decision_timeout_s=decision_timeout_s, max_decisions=max_decisions
    )


def _scenario_from_json(data: dict) -> CombatScenario:
    fields = dict(data)
    fields["enemies"] = [EnemyScenario(**e) for e in fields["enemies"]]
    return CombatScenario(**fields)


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
        "--scenario",
        type=Path,
        required=True,
        help="path to a JSON file whose keys match CombatScenario's fields "
        "(character_id, player_hp, player_max_hp, hand, draw_pile, discard_pile, "
        "enemies=[{monster_id, hp, ...}], and any optional fields)",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> EpisodeResult:
    scenario = _scenario_from_json(json.loads(args.scenario.read_text(encoding="utf-8")))
    connection = TcpConnection(host=args.host, port=args.port, connect_timeout_s=args.connect_timeout)
    async with AsyncTrainingApiClient(connection) as client:
        return await start_combat_from_state(
            client,
            scenario,
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
