"""Train a pairwise linear StableFrontierPruner from budgeted Oracle JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
_REPO_ROOT = _SRC_ROOT.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from sts2_training.decision.learned_pruner import LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION
from sts2_training.decision.pruner_features import (
    PRUNER_FEATURE_NAMES,
    PRUNER_FEATURE_SCHEMA_VERSION,
)
from sts2_training.decision.pruner_training_data import (
    PairwisePrunerExample,
    PrunerFrontierTrainingExample,
    build_pairwise_examples,
    load_pruner_frontiers,
)
from sts2_training.decision.oracle_teacher_provenance import (
    inspect_oracle_teacher_provenance,
)

DEFAULT_OUTPUT = Path("tools/output/stable_pruner_weights.json")


@dataclass(frozen=True)
class SourceSplit:
    train: list[Path]
    val: list[Path]
    test: list[Path]


@dataclass(frozen=True)
class FittedPairwiseRanker:
    scaler: Any
    model: Any

    def score(self, features: tuple[float, ...]) -> float:
        return float(
            sum(
                coefficient * (value / scale)
                for coefficient, value, scale in zip(
                    self.model.coef_[0],
                    features,
                    self.scaler.scale_,
                    strict=True,
                )
            )
        )


def split_source_files(
    paths: list[Path],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> SourceSplit:
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("val_fraction/test_fraction must each be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")

    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    n_val = round(len(shuffled) * val_fraction)
    n_test = round(len(shuffled) * test_fraction)
    return SourceSplit(
        val=shuffled[:n_val],
        test=shuffled[n_val : n_val + n_test],
        train=shuffled[n_val + n_test :],
    )


def fit_pairwise_ranker(
    pairs: list[PairwisePrunerExample],
    *,
    inverse_regularization: float,
    seed: int,
) -> FittedPairwiseRanker:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if len(pairs) < 2:
        raise ValueError("need at least two pairwise examples")
    labels = [pair.label for pair in pairs]
    if len(set(labels)) < 2:
        raise ValueError("pairwise examples must contain both labels")

    x = np.asarray([pair.features for pair in pairs], dtype=float)
    y = np.asarray(labels, dtype=int)
    sample_weight = np.asarray([pair.weight for pair in pairs], dtype=float)
    scaler = StandardScaler(with_mean=False)
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(
        C=inverse_regularization,
        fit_intercept=False,
        max_iter=1000,
        random_state=seed,
    )
    model.fit(x_scaled, y, sample_weight=sample_weight)
    return FittedPairwiseRanker(scaler=scaler, model=model)


def evaluate_ranker(
    fitted: FittedPairwiseRanker,
    frontiers: list[PrunerFrontierTrainingExample],
    *,
    min_target_gap: float,
) -> dict[str, Any]:
    pairs = build_pairwise_examples(frontiers, min_target_gap=min_target_gap)
    pair_correct = 0
    pair_weight_correct = 0.0
    pair_weight_total = 0.0
    for pair in pairs:
        score = fitted.score(pair.features)
        predicted = 1 if score >= 0.0 else 0
        if predicted == pair.label:
            pair_correct += 1
            pair_weight_correct += pair.weight
        pair_weight_total += pair.weight

    learned_recall: list[float] = []
    baseline_recall: list[float] = []
    learned_regret: list[float] = []
    baseline_regret: list[float] = []
    learned_mean_gap: list[float] = []
    baseline_mean_gap: list[float] = []

    for frontier in frontiers:
        nodes = list(frontier.nodes)
        if not nodes:
            continue
        k = min(frontier.target_beam_width, len(nodes))
        teacher = sorted(nodes, key=lambda node: node.target_value, reverse=True)[:k]
        learned = sorted(nodes, key=lambda node: fitted.score(node.features), reverse=True)[:k]
        baseline = sorted(nodes, key=lambda node: node.features[0], reverse=True)[:k]
        teacher_ids = {node.node_id for node in teacher}
        learned_ids = {node.node_id for node in learned}
        baseline_ids = {node.node_id for node in baseline}
        learned_recall.append(len(teacher_ids & learned_ids) / k)
        baseline_recall.append(len(teacher_ids & baseline_ids) / k)

        best_target = max(node.target_value for node in nodes)
        learned_regret.append(best_target - max(node.target_value for node in learned))
        baseline_regret.append(best_target - max(node.target_value for node in baseline))
        teacher_mean = sum(node.target_value for node in teacher) / k
        learned_mean_gap.append(teacher_mean - sum(node.target_value for node in learned) / k)
        baseline_mean_gap.append(teacher_mean - sum(node.target_value for node in baseline) / k)

    return {
        "frontiers": len(frontiers),
        "labeled_nodes": sum(len(frontier.nodes) for frontier in frontiers),
        "pairwise_examples": len(pairs),
        "pairwise_accuracy": None if not pairs else pair_correct / len(pairs),
        "weighted_pairwise_accuracy": (
            None if pair_weight_total == 0.0 else pair_weight_correct / pair_weight_total
        ),
        "learned_recall_at_k": _mean_or_none(learned_recall),
        "value_top_k_recall_at_k": _mean_or_none(baseline_recall),
        "learned_best_value_regret": _mean_or_none(learned_regret),
        "value_top_k_best_value_regret": _mean_or_none(baseline_regret),
        "learned_mean_selected_value_gap": _mean_or_none(learned_mean_gap),
        "value_top_k_mean_selected_value_gap": _mean_or_none(baseline_mean_gap),
    }


def weights_payload(
    fitted: FittedPairwiseRanker,
    *,
    metrics: dict[str, dict[str, Any]],
    training_files: list[Path],
    min_target_gap: float,
    terminal_weight: float,
    bootstrap_weight: float,
    oracle_teacher_provenance: dict[str, Any] | None = None,
    oracle_dataset_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model_type": "pairwise_logistic_linear_pruner",
        "artifact_schema_version": LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "training_commit": _training_commit(),
        "feature_schema_version": PRUNER_FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": hashlib.sha256(
            "\n".join(PRUNER_FEATURE_NAMES).encode("utf-8")
        ).hexdigest(),
        "feature_names": list(PRUNER_FEATURE_NAMES),
        "coefficients": fitted.model.coef_[0].tolist(),
        # Pairwise features are scaled without centering. This makes runtime node
        # scores exactly compatible with pairwise score differences.
        "scale": fitted.scaler.scale_.tolist(),
        "training": {
            "source_files": [str(path) for path in training_files],
            "min_target_gap": min_target_gap,
            "terminal_weight": terminal_weight,
            "bootstrap_weight": bootstrap_weight,
            "objective": "pairwise_logistic_ranking",
        },
        "metrics": metrics,
    }
    if oracle_teacher_provenance is not None:
        payload["oracle_teacher_provenance"] = oracle_teacher_provenance
    if oracle_dataset_provenance is not None:
        payload["oracle_dataset_provenance"] = oracle_dataset_provenance
    return payload


def _load(
    paths: list[Path],
    *,
    terminal_weight: float,
    bootstrap_weight: float,
) -> list[PrunerFrontierTrainingExample]:
    return load_pruner_frontiers(
        paths,
        terminal_weight=terminal_weight,
        bootstrap_weight=bootstrap_weight,
    )


def _mean_or_none(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _training_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inverse-regularization", type=float, default=1.0)
    parser.add_argument("--min-target-gap", type=float, default=1e-6)
    parser.add_argument("--terminal-weight", type=float, default=1.0)
    parser.add_argument("--bootstrap-weight", type=float, default=0.5)
    parser.add_argument(
        "--allow-mixed-teachers",
        action="store_true",
        help="allow Oracle v3 records from multiple teacher provenance fingerprints; "
        "the full set is recorded in the artifact",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = sorted(args.log_dir.glob("*.jsonl"))
    if not paths:
        print(f"no *.jsonl files found under {args.log_dir}", file=sys.stderr)
        return 1

    dataset_provenance = inspect_oracle_teacher_provenance(
        paths,
        allow_mixed_teachers=args.allow_mixed_teachers,
    )
    split = split_source_files(
        paths,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    training_provenance = inspect_oracle_teacher_provenance(
        split.train,
        allow_mixed_teachers=args.allow_mixed_teachers,
    )
    train_frontiers = _load(
        split.train,
        terminal_weight=args.terminal_weight,
        bootstrap_weight=args.bootstrap_weight,
    )
    train_pairs = build_pairwise_examples(
        train_frontiers,
        min_target_gap=args.min_target_gap,
    )
    fitted = fit_pairwise_ranker(
        train_pairs,
        inverse_regularization=args.inverse_regularization,
        seed=args.seed,
    )
    metrics = {
        "train": evaluate_ranker(
            fitted, train_frontiers, min_target_gap=args.min_target_gap
        ),
        "val": evaluate_ranker(
            fitted,
            _load(
                split.val,
                terminal_weight=args.terminal_weight,
                bootstrap_weight=args.bootstrap_weight,
            ),
            min_target_gap=args.min_target_gap,
        ),
        "test": evaluate_ranker(
            fitted,
            _load(
                split.test,
                terminal_weight=args.terminal_weight,
                bootstrap_weight=args.bootstrap_weight,
            ),
            min_target_gap=args.min_target_gap,
        ),
    }
    payload = weights_payload(
        fitted,
        metrics=metrics,
        training_files=split.train,
        min_target_gap=args.min_target_gap,
        terminal_weight=args.terminal_weight,
        bootstrap_weight=args.bootstrap_weight,
        oracle_teacher_provenance=training_provenance.to_json(),
        oracle_dataset_provenance=dataset_provenance.to_json(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
