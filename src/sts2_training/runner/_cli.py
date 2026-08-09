"""Shared CLI scaffolding for the `start_*` entry-point scripts - the argument set
and result printing are identical across both; only the scenario-specific
input differs (see each module's own `_parse_args`/`_run`).
"""

from __future__ import annotations

import argparse
import json
import logging
import math

from sts2_training.decision.search_modes import SEARCH_MODES
from sts2_training.runner.episode import EpisodeResult

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _port(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer between 1 and 65535") from exc
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be an integer between 1 and 65535")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=8765)
    parser.add_argument("--connect-timeout", type=_positive_float, default=5.0)
    parser.add_argument("--decision-timeout", type=_positive_float, default=30.0)
    parser.add_argument("--max-decisions", type=_positive_int, default=None)
    parser.add_argument(
        "--search-mode",
        choices=sorted(SEARCH_MODES),
        default=None,
        help="beam search preset (see decision.search_modes); default: standard",
    )
    parser.add_argument(
        "--beam-depth",
        type=_positive_int,
        default=None,
        help="override just the beam search depth of --search-mode",
    )
    parser.add_argument(
        "--log-level",
        choices=_LOG_LEVELS,
        default="WARNING",
        help="e.g. WARNING surfaces EpisodeRunner's best-effort close_instance "
        "failures, which are otherwise silent from the CLI",
    )


def configure_logging(log_level: str) -> None:
    # force=True: basicConfig() is a no-op if the root logger already has handlers
    # (e.g. a host process, or pytest's own log capture) - a CLI entry point's
    # main() calls this specifically to take ownership of its own output.
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s", force=True)


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
