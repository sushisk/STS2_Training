"""Shared CLI scaffolding for the `start_*` entry-point scripts - the argument set
and result printing are identical across all three; only the scenario-specific
input differs (see each module's own `_parse_args`/`_run`).
"""

from __future__ import annotations

import argparse
import json

from sts2_training.decision.search_modes import SEARCH_MODES
from sts2_training.runner.episode import EpisodeResult


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
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


def print_result(result: EpisodeResult) -> None:
    print(
        json.dumps(
            {
                "instance_id": result.instance_id,
                "decisions_made": result.decisions_made,
                "elapsed_s": result.elapsed_s,
                "final_dto": result.final_dto,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
