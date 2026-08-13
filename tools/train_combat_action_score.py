"""Train a supervised Combat action_score ranker from Oracle v6 root-action targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
_REPO_ROOT = _SRC_ROOT.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from sts2_training.decision.action_score_features import (
    ACTION_SCORE_FEATURE_NAMES,
    ACTION_SCORE_FEATURE_SCHEMA_VERSION,
)
from sts2_training.decision.action_score_training_data import (
    CombatActionScoreDatasetStats,
    CombatActionScoreTrainingExample,
    build_pairwise_action_score_examples,
    load_combat_action_score_examples,
)
from sts2_training.decision.learned_policy import (
    ACTION_SCORE_MODEL_TYPE,
    LEARNED_ACTION_SCORE_ARTIFACT_SCHEMA_VERSION,
)
from sts2_training.decision.oracle_log import ORACLE_RECORD_SCHEMA_VERSION, ORACLE_VALUE_MASK_VERSION
from sts2_training.decision.oracle_teacher_provenance import inspect_oracle_teacher_provenance

DEFAULT_OUTPUT = Path("tools/output/combat_action_score_weights.json")


@dataclass(frozen=True)
class SourceSplit:
    train: list[Path]
    val: list[Path]
    test: list[Path]


@dataclass(frozen=True)
class FittedActionScoreModel:
    scaler: Any
    model: Any

    def score(self, features: tuple[float, ...]) -> float:
        import numpy as np

        matrix = np.asarray([features], dtype=float)
        transformed = self.scaler.transform(matrix)
        return float(self.model.decision_function(transformed)[0])


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


def fit_action_score_model(
    examples: list[CombatActionScoreTrainingExample],
    *,
    c: float,
    tie_tolerance: float,
) -> FittedActionScoreModel:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if not math.isfinite(c) or c <= 0.0:
        raise ValueError("c must be a finite positive number")
    pairs = build_pairwise_action_score_examples(examples, tie_tolerance=tie_tolerance)
    if not pairs:
        raise ValueError("need at least one non-tied within-decision action_score pair")

    x = np.asarray([pair.feature_delta for pair in pairs], dtype=float)
    y = np.asarray([pair.label for pair in pairs], dtype=int)
    weights = np.asarray([pair.sample_weight for pair in pairs], dtype=float)
    # Pair deltas are symmetric (+d/-d), so centering is unnecessary and would not have a
    # meaningful per-action runtime analogue. Scaling only preserves the exact linear ranker.
    scaler = StandardScaler(with_mean=False)
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(
        C=c,
        fit_intercept=False,
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    )
    model.fit(x_scaled, y, sample_weight=weights)
    return FittedActionScoreModel(scaler=scaler, model=model)


def evaluate_action_score_model(
    fitted: FittedActionScoreModel,
    examples: list[CombatActionScoreTrainingExample],
    stats: CombatActionScoreDatasetStats,
    *,
    tie_tolerance: float,
) -> dict[str, Any]:
    result: dict[str, Any] = stats.to_json()
    pairs = build_pairwise_action_score_examples(examples, tie_tolerance=tie_tolerance)
    if not pairs:
        result.update(
            {
                "pair_count": 0,
                "pairwise_accuracy": None,
                "top1_decisions": 0,
                "top1_accuracy": None,
                "mean_top1_q_regret": None,
            }
        )
        return result

    correct = 0
    # Symmetric pair rows are intentional training examples; accuracy over both rows is
    # identical to accuracy over each unordered comparison and keeps weighting transparent.
    for pair in pairs:
        predicted = 1 if _score_delta(fitted, pair.feature_delta) >= 0.0 else 0
        correct += int(predicted == pair.label)

    by_decision: dict[tuple[str, str, int], list[CombatActionScoreTrainingExample]] = defaultdict(list)
    for example in examples:
        by_decision[example.decision_key].append(example)
    top1_decisions = 0
    top1_correct = 0
    regrets: list[float] = []
    for group in by_decision.values():
        if len(group) < 2:
            continue
        best_q = max(example.estimated_q for example in group)
        predicted = max(group, key=lambda example: fitted.score(example.features))
        top1_decisions += 1
        top1_correct += int(best_q - predicted.estimated_q <= tie_tolerance)
        regrets.append(max(0.0, best_q - predicted.estimated_q))

    result.update(
        {
            "pair_count": len(pairs) // 2,
            "pairwise_accuracy": correct / len(pairs),
            "top1_decisions": top1_decisions,
            "top1_accuracy": None if top1_decisions == 0 else top1_correct / top1_decisions,
            "mean_top1_q_regret": None if not regrets else sum(regrets) / len(regrets),
        }
    )
    return result


def weights_payload(
    fitted: FittedActionScoreModel,
    *,
    metrics: dict[str, dict[str, Any]],
    split: SourceSplit,
    dataset_files: list[Path],
    c: float,
    tie_tolerance: float,
    terminal_weight: float,
    bootstrap_weight: float,
    mixed_weight: float,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    training_teacher_provenance: dict[str, Any],
    dataset_teacher_provenance: dict[str, Any],
) -> dict[str, Any]:
    required_dto_version = _required_dto_version_from_metrics(metrics)
    coefficients = fitted.model.coef_[0].tolist()
    scale = fitted.scaler.scale_.tolist()
    return {
        "model_type": ACTION_SCORE_MODEL_TYPE,
        "artifact_schema_version": LEARNED_ACTION_SCORE_ARTIFACT_SCHEMA_VERSION,
        "feature_schema_version": ACTION_SCORE_FEATURE_SCHEMA_VERSION,
        "oracle_record_schema_version": ORACLE_RECORD_SCHEMA_VERSION,
        "required_mask_version": ORACLE_VALUE_MASK_VERSION,
        "required_dto_version": required_dto_version,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "training_commit": _training_commit(),
        "feature_schema_hash": hashlib.sha256(
            "\n".join(ACTION_SCORE_FEATURE_NAMES).encode("utf-8")
        ).hexdigest(),
        "feature_names": list(ACTION_SCORE_FEATURE_NAMES),
        "coefficients": coefficients,
        "intercept": 0.0,
        "mean": [0.0] * len(ACTION_SCORE_FEATURE_NAMES),
        "scale": scale,
        "training": {
            "objective": "weighted_pairwise_logistic_ranking",
            "c": c,
            "tie_tolerance": tie_tolerance,
            "terminal_weight": terminal_weight,
            "bootstrap_weight": bootstrap_weight,
            "mixed_weight": mixed_weight,
            "require_exhaustive_root_actions": True,
            "split": {
                "seed": seed,
                "val_fraction": val_fraction,
                "test_fraction": test_fraction,
                "unit": "source_file",
            },
            "input_manifest": _input_manifest(dataset_files, split=split),
        },
        "oracle_teacher_provenance": training_teacher_provenance,
        "oracle_dataset_provenance": dataset_teacher_provenance,
        "metrics": metrics,
    }


def _score_delta(fitted: FittedActionScoreModel, delta: tuple[float, ...]) -> float:
    import numpy as np

    matrix = np.asarray([delta], dtype=float)
    transformed = fitted.scaler.transform(matrix)
    return float(fitted.model.decision_function(transformed)[0])


def _required_dto_version_from_metrics(metrics: dict[str, dict[str, Any]]) -> str:
    train = metrics.get("train")
    required = train.get("dto_version") if isinstance(train, dict) else None
    if not isinstance(required, str) or not required:
        raise ValueError("training metrics must contain exact train.dto_version")
    for split_name, split_metrics in metrics.items():
        if not isinstance(split_metrics, dict):
            continue
        value = split_metrics.get("dto_version")
        if value is not None and value != required:
            raise ValueError(
                "action_score artifact metrics mix dto_version values: "
                f"train={required!r}, {split_name}={value!r}"
            )
    return required


def _input_manifest(paths: list[Path], *, split: SourceSplit) -> list[dict[str, Any]]:
    split_by_path = {
        **{path: "train" for path in split.train},
        **{path: "val" for path in split.val},
        **{path: "test" for path in split.test},
    }
    return [
        {
            "source_path": str(path),
            "sha256": _file_sha256(path),
            "split": split_by_path[path],
        }
        for path in sorted(paths, key=lambda value: str(value))
    ]


def _load(
    paths: list[Path],
    *,
    terminal_weight: float,
    bootstrap_weight: float,
    mixed_weight: float,
    allow_mixed_teachers: bool,
) -> tuple[list[CombatActionScoreTrainingExample], CombatActionScoreDatasetStats]:
    if not paths:
        return [], CombatActionScoreDatasetStats(0, 0, 0, 0, 0, 0)
    return load_combat_action_score_examples(
        paths,
        terminal_weight=terminal_weight,
        bootstrap_weight=bootstrap_weight,
        mixed_weight=mixed_weight,
        allow_mixed_teachers=allow_mixed_teachers,
        require_exhaustive_root_actions=True,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return completed.stdout.strip() or None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--tie-tolerance", type=float, default=1e-9)
    parser.add_argument("--terminal-weight", type=float, default=1.0)
    parser.add_argument("--bootstrap-weight", type=float, default=0.5)
    parser.add_argument("--mixed-weight", type=float, default=0.5)
    parser.add_argument("--allow-mixed-teachers", action="store_true")
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
    ).to_json()
    split = split_source_files(
        paths,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    if not split.train:
        raise ValueError("source-file split produced no training files")
    training_provenance = inspect_oracle_teacher_provenance(
        split.train,
        allow_mixed_teachers=args.allow_mixed_teachers,
    ).to_json()

    load_kwargs = {
        "terminal_weight": args.terminal_weight,
        "bootstrap_weight": args.bootstrap_weight,
        "mixed_weight": args.mixed_weight,
        "allow_mixed_teachers": args.allow_mixed_teachers,
    }
    train_examples, train_stats = _load(split.train, **load_kwargs)
    val_examples, val_stats = _load(split.val, **load_kwargs)
    test_examples, test_stats = _load(split.test, **load_kwargs)
    fitted = fit_action_score_model(
        train_examples,
        c=args.c,
        tie_tolerance=args.tie_tolerance,
    )
    metrics = {
        "train": evaluate_action_score_model(
            fitted, train_examples, train_stats, tie_tolerance=args.tie_tolerance
        ),
        "val": evaluate_action_score_model(
            fitted, val_examples, val_stats, tie_tolerance=args.tie_tolerance
        ),
        "test": evaluate_action_score_model(
            fitted, test_examples, test_stats, tie_tolerance=args.tie_tolerance
        ),
    }
    payload = weights_payload(
        fitted,
        metrics=metrics,
        split=split,
        dataset_files=paths,
        c=args.c,
        tie_tolerance=args.tie_tolerance,
        terminal_weight=args.terminal_weight,
        bootstrap_weight=args.bootstrap_weight,
        mixed_weight=args.mixed_weight,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        training_teacher_provenance=training_provenance,
        dataset_teacher_provenance=dataset_provenance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "metrics": metrics}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
