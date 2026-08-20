"""Compare Whole Run floor reach across baseline and learned model conditions.

The command writes a machine-readable JSON report, a compact CSV comparison, and a
human-readable Markdown table.  Each condition is evaluated with independent seeds;
the report stores the complete per-run results and the exact run/model configuration.

Example::

    python tools/evaluate_whole_run.py \
        --character-id IRONCLAD \
        --num-runs 30 \
        --search-modes standard,deep,wide \
        --beam-depth 3 \
        --output-dir data/evaluation/whole_run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.decision.learned_policy import LinearActionScorePolicy
from sts2_training.decision.learned_pruner import LinearStableFrontierPruner
from sts2_training.decision.learned_value import LinearValueModel
from sts2_training.decision.damage_race_value import DamageRaceValueFunction
from sts2_training.decision.value import HeuristicValueFunction
from sts2_training.decision.search_modes import SEARCH_MODES, resolve_search_mode
from sts2_training.runner._cli import (
    _port,
    _port_list,
    _positive_float,
    _positive_int,
    configure_logging,
)
from sts2_training.runner.beam_scope import runner_combat_beam_config
from sts2_training.runner.floor_reach_eval import FloorReachResult, run_floor_reach_eval
from sts2_training.runner.rl_server_pool import RlServerPool, free_ports

_DEFAULT_VALUE = Path("tools/output/combat_value_weights_oracle_v7_20260819.json")
_DEFAULT_POLICY = Path("tools/output/combat_action_score_weights_oracle_v7_20260819.json")
_DEFAULT_PRUNER = Path("tools/output/stable_pruner_weights_oracle_v7_20260819.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_info(path: Path, model: Any) -> dict[str, Any]:
    metadata = getattr(model, "artifact_metadata", None)
    if not isinstance(metadata, dict):
        metadata = model.oracle_provenance() if hasattr(model, "oracle_provenance") else {}
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "model_type": metadata.get("model_type"),
        "artifact_schema_version": metadata.get("artifact_schema_version"),
        "feature_schema_version": metadata.get("feature_schema_version"),
        "required_mask_version": metadata.get("required_mask_version"),
        "required_dto_version": metadata.get("required_dto_version"),
    }


def _parse_modes(raw: str) -> list[str]:
    modes = [item.strip() for item in raw.split(",") if item.strip()]
    if not modes:
        raise argparse.ArgumentTypeError("at least one search mode is required")
    unknown = sorted(set(modes) - set(SEARCH_MODES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown search mode(s): {', '.join(unknown)}; choose from {sorted(SEARCH_MODES)}"
        )
    return list(dict.fromkeys(modes))


def _parse_models(raw: str) -> list[str]:
    models = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not models:
        raise argparse.ArgumentTypeError("at least one model condition is required")
    unknown = sorted(set(models) - {"baseline", "learned"})
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown model condition(s): {', '.join(unknown)}; choose baseline or learned"
        )
    return list(dict.fromkeys(models))


def _summary(results: list[FloorReachResult]) -> dict[str, Any]:
    successful = [result for result in results if result.error is None]
    floors = [result.max_total_floor for result in successful]
    stats: dict[str, Any] | None = None
    if floors:
        mean = statistics.fmean(floors)
        stdev = statistics.pstdev(floors) if len(floors) > 1 else 0.0
        stats = {
            "mean": mean,
            "variance": stdev * stdev,
            "stdev": stdev,
            "median": statistics.median(floors),
            "min": min(floors),
            "max": max(floors),
        }
    outcomes: dict[str, int] = {}
    for result in successful:
        key = result.outcome if result.outcome is not None else "unknown"
        outcomes[key] = outcomes.get(key, 0) + 1
    return {
        "runs_requested": len(results),
        "runs_completed": len(successful),
        "runs_errored": len(results) - len(successful),
        "outcome_counts": outcomes,
        "floor_stats": stats,
        "errors": [{"run_id": r.run_id, "error": r.error} for r in results if r.error],
    }


def _flat_row(condition: dict[str, Any]) -> dict[str, Any]:
    stats = condition["summary"].get("floor_stats") or {}
    return {
        "condition": condition["condition"],
        "model": condition["model"],
        "ascension": condition["run_config"]["ascension"],
        "search_mode": condition["run_config"]["search_mode"],
        "beam_depth": condition["run_config"]["beam_depth"],
        "beam_width": condition["run_config"]["beam_width"],
        "top_k_actions": condition["run_config"]["top_k_actions"],
        "pruner": condition["run_config"]["pruner"],
        "board_score": condition["run_config"]["board_score"],
        "num_runs": condition["run_config"]["num_runs"],
        "max_decisions": condition["run_config"]["max_decisions"],
        "runs_completed": condition["summary"]["runs_completed"],
        "runs_errored": condition["summary"]["runs_errored"],
        "mean_floor": stats.get("mean"),
        "variance": stats.get("variance"),
        "stdev": stats.get("stdev"),
        "median_floor": stats.get("median"),
        "min_floor": stats.get("min"),
        "max_floor": stats.get("max"),
    }


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        ("condition", "Condition"),
        ("model", "Model"),
        ("ascension", "Ascension"),
        ("search_mode", "Search"),
        ("beam_depth", "Depth"),
        ("beam_width", "Width"),
        ("top_k_actions", "Top-k"),
        ("pruner", "Pruner"),
        ("board_score", "Board score"),
        ("runs_completed", "Runs"),
        ("max_decisions", "Max decisions"),
        ("mean_floor", "Mean floor"),
        ("variance", "Variance"),
        ("stdev", "Std dev"),
        ("median_floor", "Median"),
        ("min_floor", "Min"),
        ("max_floor", "Max"),
    ]
    lines = [
        "# Whole Run evaluation",
        "",
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            values.append("" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mode_configs(args: argparse.Namespace) -> dict[str, BeamSearchConfig]:
    """Resolve one BeamSearchConfig per selected search mode.

    These configs exist only to carry --beam-width/--top-k-actions/--beam-depth overrides;
    the semantic scope is always the full Combat domain. `runner_combat_beam_config` is
    applied here so that intent is explicit at the construction site rather than depending
    on what the receiving runner does with a caller-supplied config.
    """

    return {
        mode: runner_combat_beam_config(
            replace(
                resolve_search_mode(mode, max_depth=args.beam_depth),
                turn_aligned_leaves=args.turn_aligned_leaves,
                turn_boundary_scoring=args.turn_boundary_scoring,
                **{
                    **({"beam_width": args.beam_width} if args.beam_width is not None else {}),
                    **(
                        {"top_k_actions": args.top_k_actions}
                        if args.top_k_actions is not None
                        else {}
                    ),
                },
            )
        )
        for mode in _parse_modes(args.search_modes)
    }


async def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    modes = _parse_modes(args.search_modes)
    model_names = _parse_models(args.models)
    mode_configs = _mode_configs(args)

    value_path = args.value_weights.resolve()
    policy_path = args.action_score_weights.resolve()
    pruner_path = args.stable_pruner_weights.resolve()
    for path in (value_path, policy_path, pruner_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.board_score == "heuristic":
        learned_value: Any = HeuristicValueFunction()
    elif args.board_score == "damage_race":
        learned_value = DamageRaceValueFunction()
    else:
        learned_value = LinearValueModel.from_weights_file(value_path)
    learned_policy = LinearActionScorePolicy.from_weights_file(policy_path)
    learned_pruner = LinearStableFrontierPruner.from_weights_file(pruner_path)
    artifacts = {
        "value": (
            {"path": None, **dict(learned_value.oracle_provenance())}
            if args.board_score in ("heuristic", "damage_race")
            else _artifact_info(value_path, learned_value)
        ),
        "action_score": _artifact_info(policy_path, learned_policy),
        "stable_pruner": _artifact_info(pruner_path, learned_pruner),
    }

    conditions: list[dict[str, Any]] = []
    model_objects = {
        "baseline": (None, None, None),
        "learned": (
            learned_policy,
            learned_value,
            learned_pruner if args.learned_pruner == "learned" else None,
        ),
    }
    for model_name in model_names:
        models = model_objects[model_name]
        for mode in modes:
            results = await run_floor_reach_eval(
                character_id=args.character_id,
                num_runs=args.num_runs,
                concurrency=args.concurrency,
                ascension=args.ascension,
                use_beam=mode != "none",
                host=args.host,
                port=args.port,
                connect_timeout_s=args.connect_timeout,
                decision_timeout_s=args.decision_timeout,
                max_decisions=args.max_decisions,
                search_mode=mode_configs[mode],
                beam_max_depth=None,
                policy=models[0],
                value_fn=models[1],
                stable_pruner=models[2],
                detailed_log_dir=args.detailed_log_dir,
                eval_epsilon=args.eval_epsilon,
                ports=args.ports,
            )
            conditions.append({
                "condition": f"{model_name}:{mode}:pruner={args.learned_pruner if model_name == 'learned' else 'baseline'}",
                "model": model_name,
                "run_config": {
                    "character_id": args.character_id,
                    "ascension": args.ascension,
                    "search_mode": mode,
                    "beam_depth": mode_configs[mode].max_depth,
                    "beam_width": mode_configs[mode].beam_width,
                    "top_k_actions": mode_configs[mode].top_k_actions,
                    "pruner": args.learned_pruner if model_name == "learned" else "baseline",
                    "board_score": args.board_score if model_name == "learned" else "baseline",
                    "turn_aligned_leaves": args.turn_aligned_leaves,
                    "turn_boundary_scoring": args.turn_boundary_scoring,
                    "num_runs": args.num_runs,
                    "concurrency": args.concurrency,
                    "max_decisions": args.max_decisions,
                    "decision_timeout_s": args.decision_timeout,
                    "host": args.host,
                    "port": args.port,
                    "ports": args.ports,
                },
                "summary": _summary(results),
                "runs": [vars(result) for result in results],
            })

    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifacts": artifacts,
        "evaluation_config": {
            "character_id": args.character_id,
            "ascension": args.ascension,
            "search_modes": modes,
            "beam_depth": args.beam_depth,
            "beam_width": args.beam_width,
            "top_k_actions": args.top_k_actions,
            "num_runs_per_condition": args.num_runs,
            "conditions": model_names,
            "learned_pruner": args.learned_pruner,
            "board_score": args.board_score,
            "overrides": {
                "beam_depth": args.beam_depth,
                "beam_width": args.beam_width,
                "top_k_actions": args.top_k_actions,
            },
        },
        "conditions": conditions,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=8765)
    parser.add_argument("--connect-timeout", type=_positive_float, default=5.0)
    parser.add_argument("--decision-timeout", type=_positive_float, default=90.0)
    parser.add_argument("--max-decisions", type=_positive_int, default=1000)
    parser.add_argument(
        "--beam-depth",
        type=_positive_int,
        default=None,
        help="override Beam Search depth for every selected search mode",
    )
    parser.add_argument(
        "--beam-width",
        type=_positive_int,
        default=None,
        help="override Beam Search width for every selected search mode",
    )
    parser.add_argument(
        "--top-k-actions",
        type=_positive_int,
        default=None,
        help="override the number of policy candidates proposed at each Beam step",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="WARNING",
    )
    parser.add_argument("--character-id", required=True)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument(
        "--turn-aligned-leaves",
        action="store_true",
        help=(
            "score each leaf after its forced End Turn when the depth budget stopped it "
            "earlier in the turn cycle than its siblings, so root actions are not "
            "compared across different amounts of unpaid enemy damage"
        ),
    )
    parser.add_argument(
        "--turn-boundary-scoring",
        action="store_true",
        help=(
            "keep the whole search inside one player turn (End Turn is never expanded) "
            "and score every leaf, including the do-nothing root, after a forced End "
            "Turn; supersedes --turn-aligned-leaves and cannot be combined with it"
        ),
    )
    parser.add_argument(
        "--eval-epsilon",
        type=float,
        default=0.0,
        help=(
            "epsilon-greedy exploration rate for the non-Combat fallback selector; "
            "0.0 (default) measures the policy itself"
        ),
    )
    parser.add_argument("--num-runs", type=_positive_int, default=30)
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=None,
        help="parallel runs; defaults to the number of --ports (1 without them)",
    )
    parser.add_argument(
        "--start-rl-servers",
        type=_positive_int,
        default=None,
        metavar="N",
        help=(
            "start N paired STS2_RL servers (python -m API.tcp_server) for this run and "
            "shut them down afterwards. Ports are chosen automatically unless --ports is "
            "given. Requires --rl-root or STS2_RL_ROOT"
        ),
    )
    parser.add_argument(
        "--rl-root",
        type=Path,
        default=None,
        help="paired STS2_RL checkout for --start-rl-servers (default: $STS2_RL_ROOT)",
    )
    parser.add_argument(
        "--rl-startup-timeout",
        type=_positive_float,
        default=180.0,
        help="seconds to wait for each started RL server to accept connections",
    )
    parser.add_argument(
        "--ports",
        type=_port_list,
        default=None,
        help=(
            "comma-separated RL server ports, one per worker (e.g. 8765,8766,8767). "
            "One RL server serializes every request behind a single lock and hosts the "
            "Emulator in one CLR process, so parallelism needs one server per port"
        ),
    )
    parser.add_argument("--search-modes", default="standard,deep,wide")
    parser.add_argument(
        "--models",
        default="baseline,learned",
        help="comma-separated conditions: baseline, learned",
    )
    parser.add_argument(
        "--learned-pruner",
        choices=("learned", "baseline"),
        default="learned",
        help="pruner used for the learned condition; baseline selects ValueTopKPruner",
    )
    parser.add_argument(
        "--board-score",
        choices=("damage_race", "heuristic", "learned"),
        default="damage_race",
        help=(
            "board score used by the learned condition. `damage_race` is the default: it "
            "scores a state as the HP the player is projected to end the combat with, so "
            "states at different points in the turn cycle stay comparable"
        ),
    )
    parser.add_argument("--value-weights", type=Path, default=_DEFAULT_VALUE)
    parser.add_argument("--action-score-weights", type=Path, default=_DEFAULT_POLICY)
    parser.add_argument("--stable-pruner-weights", type=Path, default=_DEFAULT_PRUNER)
    parser.add_argument("--output-dir", type=Path, default=Path("data/evaluation/whole_run"))
    parser.add_argument(
        "--detailed-log-dir",
        type=Path,
        default=None,
        help=(
            "write one per-run JSONL stream containing root board snapshots, action scores, "
            "selection events, and beam score traces (omitted by default)"
        ),
    )
    return parser.parse_args(argv)


def _run_report(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate, starting and stopping the RL servers when asked to manage them.

    `--start-rl-servers N` exists because a sharded evaluation is otherwise a manual
    choreography of N terminals, and a forgotten server turns one worker's whole share of
    the runs into connection errors. The pool also guarantees the servers (and their
    Emulator worker processes) are gone afterwards, including on Ctrl-C.
    """

    import asyncio

    if not args.start_rl_servers:
        return asyncio.run(_evaluate(args))

    ports = args.ports or free_ports(args.start_rl_servers, host=args.host)
    if len(ports) != args.start_rl_servers:
        raise SystemExit(
            f"--start-rl-servers {args.start_rl_servers} does not match the "
            f"{len(ports)} port(s) given by --ports"
        )
    with RlServerPool(
        ports=ports,
        root=args.rl_root,
        host=args.host,
        log_dir=args.output_dir,
        startup_timeout_s=args.rl_startup_timeout,
    ) as started_ports:
        args.ports = started_ports
        return asyncio.run(_evaluate(args))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging(args.log_level)
    report = _run_report(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    stem = output_dir / f"whole_run_eval-{stamp}-{args.character_id.lower()}"
    json_path = stem.with_suffix(".json")
    csv_path = stem.with_suffix(".csv")
    md_path = stem.with_suffix(".md")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    rows = [_flat_row(condition) for condition in report["conditions"]]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(md_path, rows)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path), "rows": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
