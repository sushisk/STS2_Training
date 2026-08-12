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
    load_pruner_frontiers,
)


def evaluate_artifact(
    pruner: LinearStableFrontierPruner,
    frontiers: list[PrunerFrontierTrainingExample],
) -> dict[str, Any]:
    learned_recall: list[float] = []
    baseline_recall: list[float] = []
    learned_regret: list[float] = []
    baseline_regret: list[float] = []
    learned_mean_gap: list[float] = []
    baseline_mean_gap: list[float] = []

    pair_correct = 0
    pair_total = 0
    weighted_correct = 0.0
    weighted_total = 0.0

    for frontier in frontiers:
        nodes = list(frontier.nodes)
        if len(nodes) < 2:
            continue

        for left_index in range(len(nodes)):
            for right_index in range(left_index + 1, len(nodes)):
                left = nodes[left_index]
                right = nodes[right_index]
                if left.target_value == right.target_value:
                    continue
                left_score = pruner.score_features(left.features)
                right_score = pruner.score_features(right.features)
                predicted_left_better = left_score >= right_score
                actual_left_better = left.target_value > right.target_value
                weight = min(left.target_weight, right.target_weight)
                pair_total += 1
                weighted_total += weight
                if predicted_left_better == actual_left_better:
                    pair_correct += 1
                    weighted_correct += weight

        k = min(frontier.target_beam_width, len(nodes))
        teacher = sorted(nodes, key=lambda node: node.target_value, reverse=True)[:k]
        learned = sorted(
            nodes,
            key=lambda node: pruner.score_features(node.features),
            reverse=True,
        )[:k]
        baseline = sorted(nodes, key=lambda node: node.features[0], reverse=True)[:k]

        teacher_ids = {node.node_id for node in teacher}
        learned_recall.append(len(teacher_ids & {node.node_id for node in learned}) / k)
        baseline_recall.append(len(teacher_ids & {node.node_id for node in baseline}) / k)

        best_target = max(node.target_value for node in nodes)
        learned_regret.append(best_target - max(node.target_value for node in learned))
        baseline_regret.append(best_target - max(node.target_value for node in baseline))
        teacher_mean = sum(node.target_value for node in teacher) / k
        learned_mean_gap.append(teacher_mean - sum(node.target_value for node in learned) / k)
        baseline_mean_gap.append(teacher_mean - sum(node.target_value for node in baseline) / k)

    return {
        "artifact_version": pruner.version,
        "frontiers": len(frontiers),
        "labeled_nodes": sum(len(frontier.nodes) for frontier in frontiers),
        "pairwise_comparisons": pair_total,
        "pairwise_accuracy": None if pair_total == 0 else pair_correct / pair_total,
        "weighted_pairwise_accuracy": (
            None if weighted_total == 0.0 else weighted_correct / weighted_total
        ),
        "learned_recall_at_k": _mean_or_none(learned_recall),
        "value_top_k_recall_at_k": _mean_or_none(baseline_recall),
        "learned_best_value_regret": _mean_or_none(learned_regret),
        "value_top_k_best_value_regret": _mean_or_none(baseline_regret),
        "learned_mean_selected_value_gap": _mean_or_none(learned_mean_gap),
        "value_top_k_mean_selected_value_gap": _mean_or_none(baseline_mean_gap),
    }


def _mean_or_none(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--terminal-weight", type=float, default=1.0)
    parser.add_argument("--bootstrap-weight", type=float, default=0.5)
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
    frontiers = load_pruner_frontiers(
        paths,
        terminal_weight=args.terminal_weight,
        bootstrap_weight=args.bootstrap_weight,
    )
    metrics = evaluate_artifact(pruner, frontiers)
    metrics["teacher_provenance_match"] = provenance_match
    metrics["training_oracle_teacher_provenance"] = training_provenance
    metrics["evaluation_oracle_teacher_provenance"] = evaluation_provenance.to_json()
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
