"""Fit board_eval's Run Win/Lose logistic model and write JSON weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import subprocess
import sys
from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
_REPO_ROOT = _SRC_ROOT.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from sts2_training.board_eval.card_features import DEFAULT_CARD_FEATURES_CSV, CardFeatureExtractor
from sts2_training.board_eval.dto_adapter import UnknownCardPolicy
from sts2_training.board_eval.training_data import (
    MODEL_FEATURE_NAMES,
    BoardStateExample,
    build_examples_from_log,
    iter_log_events,
    label_from_events,
)

_LOG = logging.getLogger(__name__)
DEFAULT_OUTPUT = Path("tools/output/board_eval_model_weights.json")
_ARTIFACT_SCHEMA_VERSION = 1
_FEATURE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunSplit:
    train: list[Path]
    val: list[Path]
    test: list[Path]


def split_logs_by_run(
    log_paths: list[Path],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> RunSplit:
    """Split whole Run files, stratified by terminal Win/Lose label."""
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("val_fraction/test_fraction must each be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")

    groups: dict[int | None, list[Path]] = {0: [], 1: [], None: []}
    for log_path in log_paths:
        label = label_from_events(list(iter_log_events(log_path)))
        groups[label].append(log_path)

    rng = random.Random(seed)
    split_parts: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for label in (0, 1, None):
        group = list(groups[label])
        rng.shuffle(group)
        n_val = round(len(group) * val_fraction)
        n_test = round(len(group) * test_fraction)
        split_parts["val"].extend(group[:n_val])
        split_parts["test"].extend(group[n_val : n_val + n_test])
        split_parts["train"].extend(group[n_val + n_test :])

    for values in split_parts.values():
        rng.shuffle(values)

    if groups[None]:
        _LOG.info(
            "split includes %d unlabeled Run log(s); they yield no training examples",
            len(groups[None]),
        )
    return RunSplit(
        train=split_parts["train"],
        val=split_parts["val"],
        test=split_parts["test"],
    )


def _examples_for(
    log_paths: list[Path],
    extractor: CardFeatureExtractor,
    *,
    on_unknown_card: UnknownCardPolicy,
    state_kinds: Collection[str] | None = None,
) -> list[BoardStateExample]:
    examples: list[BoardStateExample] = []
    for log_path in log_paths:
        examples.extend(
            build_examples_from_log(
                log_path,
                extractor,
                on_unknown_card=on_unknown_card,
                state_kinds=state_kinds,
            )
        )
    return examples


def _matrix(examples: list[BoardStateExample]) -> tuple[Any, Any]:
    import numpy as np

    x = np.array([example.to_model_vector() for example in examples], dtype=float)
    y = np.array([example.label for example in examples], dtype=int)
    return x, y


def fit_model(
    train_examples: list[BoardStateExample],
    *,
    inverse_regularization: float,
    seed: int,
) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(train_examples) < 2:
        raise ValueError("need at least 2 training examples to fit a model")
    x_train, y_train = _matrix(train_examples)
    if len(set(y_train.tolist())) < 2:
        raise ValueError(
            "training examples contain only one label value (all Win or all Lose) - "
            "LogisticRegression needs both classes represented"
        )
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=inverse_regularization, max_iter=1000, random_state=seed),
    )
    pipeline.fit(x_train, y_train)
    return pipeline


def evaluate_model(pipeline: Any, examples: list[BoardStateExample]) -> dict[str, Any]:
    if not examples:
        return {"count": 0}

    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

    x, y = _matrix(examples)
    probabilities = pipeline.predict_proba(x)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics: dict[str, Any] = {
        "count": len(examples),
        "accuracy": float(accuracy_score(y, predictions)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "calibration": _calibration_table(y, probabilities),
    }
    single_class = len(set(y.tolist())) < 2
    if single_class:
        _LOG.info(
            "evaluation split contains only one class; "
            "log_loss and roc_auc are undefined and set to null"
        )
    metrics["log_loss"] = None if single_class else float(log_loss(y, probabilities))
    metrics["roc_auc"] = None if single_class else float(roc_auc_score(y, probabilities))
    return metrics


def _calibration_table(
    y: Any,
    probabilities: Any,
    *,
    num_buckets: int = 10,
) -> list[dict[str, Any]]:
    buckets: list[dict[str, Any]] = []
    for bucket in range(num_buckets):
        low, high = bucket / num_buckets, (bucket + 1) / num_buckets
        in_bucket = (probabilities >= low) & (
            probabilities < high if bucket < num_buckets - 1 else probabilities <= high
        )
        count = int(in_bucket.sum())
        buckets.append(
            {
                "predicted_range": [low, high],
                "count": count,
                "mean_predicted": float(probabilities[in_bucket].mean()) if count else None,
                "observed_win_rate": float(y[in_bucket].mean()) if count else None,
            }
        )
    return buckets


def weights_payload(
    pipeline: Any,
    metrics_by_split: dict[str, dict[str, Any]],
    *,
    card_catalog_path: Path | str = DEFAULT_CARD_FEATURES_CSV,
    training_state_kinds: Collection[str] | None = None,
) -> dict[str, Any]:
    scaler = pipeline.named_steps["standardscaler"]
    model = pipeline.named_steps["logisticregression"]
    catalog_path = Path(card_catalog_path)
    state_kind_scope = (
        None if training_state_kinds is None else sorted(set(training_state_kinds))
    )
    return {
        "model_type": "logistic_regression",
        "artifact_schema_version": _ARTIFACT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "card_catalog_hash": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "card_catalog_path": str(catalog_path),
        "training_commit": _training_commit(),
        "training_state_kinds": state_kind_scope,
        "feature_schema_version": _FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": hashlib.sha256(
            "\n".join(MODEL_FEATURE_NAMES).encode("utf-8")
        ).hexdigest(),
        "feature_names": list(MODEL_FEATURE_NAMES),
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "metrics": metrics_by_split,
    }


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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--card-csv", type=Path, default=DEFAULT_CARD_FEATURES_CSV)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inverse-regularization", type=float, default=1.0)
    parser.add_argument("--on-unknown-card", choices=["raise", "skip"], default="raise")
    parser.add_argument(
        "--state-kind",
        action="append",
        dest="state_kinds",
        help="optional state_kind filter; repeat to include multiple kinds",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    log_paths = sorted(args.log_dir.glob("*.jsonl"))
    if not log_paths:
        print(f"no *.jsonl logs found under {args.log_dir}", file=sys.stderr)
        return 1

    split = split_logs_by_run(
        log_paths,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    extractor = CardFeatureExtractor.from_csv(args.card_csv)
    kwargs = {"on_unknown_card": args.on_unknown_card, "state_kinds": args.state_kinds}
    train_examples = _examples_for(split.train, extractor, **kwargs)
    val_examples = _examples_for(split.val, extractor, **kwargs)
    test_examples = _examples_for(split.test, extractor, **kwargs)

    print(
        f"Runs: {len(split.train)} train / {len(split.val)} val / {len(split.test)} test "
        f"-> examples: {len(train_examples)} / {len(val_examples)} / {len(test_examples)}"
    )
    pipeline = fit_model(
        train_examples,
        inverse_regularization=args.inverse_regularization,
        seed=args.seed,
    )
    metrics_by_split = {
        "train": evaluate_model(pipeline, train_examples),
        "val": evaluate_model(pipeline, val_examples),
        "test": evaluate_model(pipeline, test_examples),
    }
    for split_name, metrics in metrics_by_split.items():
        print(
            f"{split_name}: "
            f"{json.dumps({k: v for k, v in metrics.items() if k != 'calibration'})}"
        )

    payload = weights_payload(
        pipeline,
        metrics_by_split,
        card_catalog_path=args.card_csv,
        training_state_kinds=args.state_kinds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
