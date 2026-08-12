"""Evaluate a trained stable-pruner artifact on held-out Oracle JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from sts2_training.decision.learned_pruner import LinearStableFrontierPruner
from sts2_training.decision.oracle_teacher_provenance import (
    inspect_oracle_teacher_provenance,
    require_matching_teacher_provenance,
)
from sts2_training.decision.pruner_training_data import (
    PrunerFrontierTrainingExample,
    build_pairwise_examples,
    evaluate_selection_metrics,
    load_pruner_frontiers,
)

_DEFAULT_MIN_TARGET_GAP = 1e-6
_DEFAULT_TERMINAL_WEIGHT = 1.0
_DEFAULT_BOOTSTRAP_WEIGHT = 0.5


def evaluate_artifact(
    pruner: LinearStableFrontierPruner,
    frontiers: list[PrunerFrontierTrainingExample],
    *,
    min_target_gap: float = _DEFAULT_MIN_TARGET_GAP,
) -> dict[str, Any]:
    pairs = build_pairwise_examples(frontiers, min_target_gap=min_target_gap)
    pair_correct = 0
    weighted_correct = 0.0
    weighted_total = 0.0
    for pair in pairs:
        score = pruner.score_features(pair.features)
        predicted = 1 if score >= 0.0 else 0
        if predicted == pair.label:
            pair_correct += 1
            weighted_correct += pair.weight
        weighted_total += pair.weight

    metrics = evaluate_selection_metrics(
        frontiers,
        score_node=lambda node: pruner.score_features(node.features),
    )
    metadata = pruner.artifact_metadata
    metrics.update(
        {
            "artifact_version": pruner.version,
            "artifact_sha256": metadata.get("artifact_sha256"),
            "artifact_schema_version": metadata.get("artifact_schema_version"),
            "stable_prune_node_view_schema_version": metadata.get(
                "stable_prune_node_view_schema_version"
            ),
            "feature_schema_version": metadata.get("feature_schema_version"),
            "oracle_record_schema_version": metadata.get("oracle_record_schema_version"),
            "pairwise_comparisons": len(pairs),
            "pairwise_accuracy": None if not pairs else pair_correct / len(pairs),
            "weighted_pairwise_accuracy": (
                None if weighted_total == 0.0 else weighted_correct / weighted_total
            ),
        }
    )
    return metrics


def resolve_evaluation_config(
    pruner: LinearStableFrontierPruner,
    *,
    min_target_gap: float | None,
    terminal_weight: float | None,
    bootstrap_weight: float | None,
) -> dict[str, Any]:
    """Resolve held-out metric semantics, preferring the artifact training contract."""

    metadata = pruner.artifact_metadata
    raw_training = metadata.get("training")
    training = raw_training if isinstance(raw_training, Mapping) else {}

    resolved_gap, gap_source = _resolve_number(
        override=min_target_gap,
        training=training,
        key="min_target_gap",
        fallback=_DEFAULT_MIN_TARGET_GAP,
        allow_zero=True,
    )
    resolved_terminal, terminal_source = _resolve_number(
        override=terminal_weight,
        training=training,
        key="terminal_weight",
        fallback=_DEFAULT_TERMINAL_WEIGHT,
        allow_zero=False,
    )
    resolved_bootstrap, bootstrap_source = _resolve_number(
        override=bootstrap_weight,
        training=training,
        key="bootstrap_weight",
        fallback=_DEFAULT_BOOTSTRAP_WEIGHT,
        allow_zero=False,
    )
    return {
        "min_target_gap": resolved_gap,
        "terminal_weight": resolved_terminal,
        "bootstrap_weight": resolved_bootstrap,
        "sources": {
            "min_target_gap": gap_source,
            "terminal_weight": terminal_source,
            "bootstrap_weight": bootstrap_source,
        },
    }


def _resolve_number(
    *,
    override: float | None,
    training: Mapping[str, Any],
    key: str,
    fallback: float,
    allow_zero: bool,
) -> tuple[float, str]:
    if override is not None:
        value = float(override)
        _validate_config_number(key, value, allow_zero=allow_zero)
        return value, "cli_override"
    training_value = training.get(key)
    if not isinstance(training_value, bool) and isinstance(training_value, (int, float)):
        value = float(training_value)
        _validate_config_number(key, value, allow_zero=allow_zero)
        return value, "artifact_training"
    return fallback, "evaluator_default"


def _validate_config_number(key: str, value: float, *, allow_zero: bool) -> None:
    if allow_zero:
        if value < 0.0:
            raise ValueError(f"{key} must be non-negative")
    elif value <= 0.0:
        raise ValueError(f"{key} must be positive")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument(
        "--min-target-gap",
        type=float,
        default=None,
        help="override artifact training min_target_gap for diagnostic evaluation",
    )
    parser.add_argument(
        "--terminal-weight",
        type=float,
        default=None,
        help="override artifact training terminal target weight",
    )
    parser.add_argument(
        "--bootstrap-weight",
        type=float,
        default=None,
        help="override artifact training value-bootstrap target weight",
    )
    parser.add_argument(
        "--allow-mixed-teachers",
        action="store_true",
        help="allow multiple teacher provenance fingerprints inside the held-out dataset",
    )
    parser.add_argument(
        "--allow-teacher-mismatch",
        action="store_true",
        help="evaluate even when held-out teacher provenance differs from the artifact; "
        "both provenance sets remain recorded in the report",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = sorted(args.log_dir.glob("*.jsonl"))
    if not paths:
        print(f"no *.jsonl files found under {args.log_dir}", file=sys.stderr)
        return 1

    evaluation_provenance = inspect_oracle_teacher_provenance(
        paths,
        allow_mixed_teachers=args.allow_mixed_teachers,
    )
    pruner = LinearStableFrontierPruner.from_weights_file(args.weights)
    raw_training_provenance = pruner.artifact_metadata.get("oracle_teacher_provenance")
    training_provenance = (
        raw_training_provenance if isinstance(raw_training_provenance, Mapping) else None
    )
    provenance_match = require_matching_teacher_provenance(
        training_provenance,
        evaluation_provenance,
        allow_teacher_mismatch=args.allow_teacher_mismatch,
    )
    evaluation_config = resolve_evaluation_config(
        pruner,
        min_target_gap=args.min_target_gap,
        terminal_weight=args.terminal_weight,
        bootstrap_weight=args.bootstrap_weight,
    )
    frontiers = load_pruner_frontiers(
        paths,
        terminal_weight=evaluation_config["terminal_weight"],
        bootstrap_weight=evaluation_config["bootstrap_weight"],
        allow_mixed_teachers=args.allow_mixed_teachers,
    )
    metrics = evaluate_artifact(
        pruner,
        frontiers,
        min_target_gap=evaluation_config["min_target_gap"],
    )
    metrics["evaluation_config"] = evaluation_config
    metrics["teacher_provenance_match"] = provenance_match
    metrics["training_oracle_teacher_provenance"] = training_provenance
    metrics["evaluation_oracle_teacher_provenance"] = evaluation_provenance.to_json()
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
