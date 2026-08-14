"""Train a supervised Combat ValueModel from Oracle v6 root-value samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from sts2_training.decision.learned_value import LEARNED_VALUE_ARTIFACT_SCHEMA_VERSION
from sts2_training.decision.oracle_log import (
    ORACLE_RECORD_SCHEMA_VERSION,
    ORACLE_VALUE_MASK_VERSION,
)
from sts2_training.decision.oracle_teacher_provenance import inspect_oracle_teacher_provenance
from sts2_training.decision.value_features import VALUE_FEATURE_NAMES, VALUE_FEATURE_SCHEMA_VERSION
from sts2_training.decision.value_training_data import (
    CombatValueDatasetStats,
    CombatValueTrainingExample,
    load_combat_value_examples,
)

DEFAULT_OUTPUT = Path("tools/output/combat_value_weights.json")


@dataclass(frozen=True)
class SourceSplit:
    train: list[Path]
    val: list[Path]
    test: list[Path]


@dataclass(frozen=True)
class FittedValueModel:
    scaler: Any
    model: Any

    def predict(self, features: tuple[float, ...]) -> float:
        import numpy as np

        matrix = np.asarray([features], dtype=float)
        return float(self.model.predict(self.scaler.transform(matrix))[0])


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


def fit_value_model(
    examples: list[CombatValueTrainingExample],
    *,
    alpha: float,
) -> FittedValueModel:
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    if not examples:
        raise ValueError("need at least one usable Combat Value training example")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be a finite non-negative number")
    x = np.asarray([example.features for example in examples], dtype=float)
    y = np.asarray([example.target_value for example in examples], dtype=float)
    weights = np.asarray([example.sample_weight for example in examples], dtype=float)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = Ridge(alpha=alpha)
    model.fit(x_scaled, y, sample_weight=weights)
    return FittedValueModel(scaler=scaler, model=model)


def evaluate_value_model(
    fitted: FittedValueModel,
    examples: list[CombatValueTrainingExample],
    stats: CombatValueDatasetStats,
) -> dict[str, Any]:
    result: dict[str, Any] = stats.to_json()
    if not examples:
        result.update(
            {
                "mae": None,
                "rmse": None,
                "weighted_mae": None,
                "weighted_rmse": None,
                "r2": None,
            }
        )
        return result

    predictions = [fitted.predict(example.features) for example in examples]
    errors = [
        prediction - example.target_value
        for prediction, example in zip(predictions, examples, strict=True)
    ]
    weights = [example.sample_weight for example in examples]
    weight_total = sum(weights)
    result.update(
        {
            "mae": sum(abs(error) for error in errors) / len(errors),
            "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
            "weighted_mae": (
                None
                if weight_total == 0.0
                else sum(
                    weight * abs(error)
                    for weight, error in zip(weights, errors, strict=True)
                )
                / weight_total
            ),
            "weighted_rmse": (
                None
                if weight_total == 0.0
                else math.sqrt(
                    sum(
                        weight * error * error
                        for weight, error in zip(weights, errors, strict=True)
                    )
                    / weight_total
                )
            ),
            "r2": _r2_or_none(
                [example.target_value for example in examples],
                predictions,
            ),
        }
    )
    return result


def derive_terminal_values(
    examples: list[CombatValueTrainingExample],
    teacher_provenance: dict[str, Any] | None = None,
) -> dict[str, float]:
    by_outcome: dict[str, list[float]] = {"victory": [], "defeat": []}
    for example in examples:
        if example.target_source != "terminal":
            continue
        outcome = _terminal_outcome(example.masked_emulator_dto)
        if outcome is not None:
            by_outcome[outcome].append(example.target_value)

    result: dict[str, float] = _terminal_values_from_teacher_provenance(
        teacher_provenance
    )
    for outcome, values in by_outcome.items():
        if not values:
            continue
        reference = values[0]
        tolerance = max(1e-9, abs(reference) * 1e-12)
        if any(abs(value - reference) > tolerance for value in values[1:]):
            raise ValueError(
                f"terminal Oracle targets for {outcome!r} are inconsistent; "
                "cannot advertise exact_terminal_utility"
            )
        existing = result.get(outcome)
        if existing is not None and abs(existing - reference) > tolerance:
            raise ValueError(
                f"terminal Oracle target for {outcome!r} conflicts with teacher provenance"
            )
        result[outcome] = float(reference)
    return result


def _terminal_values_from_teacher_provenance(
    summary: dict[str, Any] | None,
) -> dict[str, float]:
    if not isinstance(summary, dict):
        return {}
    candidates: dict[str, list[float]] = {"victory": [], "defeat": []}
    for teacher in summary.get("teachers") or []:
        if not isinstance(teacher, dict):
            continue
        provenance = teacher.get("provenance")
        if not isinstance(provenance, dict):
            continue
        metadata = provenance.get("teacher_value_metadata")
        if not isinstance(metadata, dict):
            continue
        explicit = metadata.get("terminal_values")
        if isinstance(explicit, dict):
            for outcome in ("victory", "defeat"):
                raw = explicit.get(outcome)
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    candidates[outcome].append(float(raw))
        weights = metadata.get("weights")
        if isinstance(weights, dict):
            for outcome, field in (
                ("victory", "victory_bonus"),
                ("defeat", "defeat_penalty"),
            ):
                raw = weights.get(field)
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    candidates[outcome].append(float(raw))
    result: dict[str, float] = {}
    for outcome, values in candidates.items():
        if not values:
            continue
        reference = values[0]
        tolerance = max(1e-9, abs(reference) * 1e-12)
        if any(abs(value - reference) > tolerance for value in values[1:]):
            raise ValueError(
                f"teacher provenance has inconsistent {outcome!r} terminal utilities"
            )
        result[outcome] = reference
    return result


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
                "Value artifact metrics mix dto_version values: "
                f"train={required!r}, {split_name}={value!r}"
            )
    return required


def weights_payload(
    fitted: FittedValueModel,
    *,
    metrics: dict[str, dict[str, Any]],
    split: SourceSplit,
    dataset_files: list[Path],
    alpha: float,
    terminal_weight: float,
    bootstrap_weight: float,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    training_teacher_provenance: dict[str, Any],
    dataset_teacher_provenance: dict[str, Any],
    terminal_values: dict[str, float],
) -> dict[str, Any]:
    scaler = fitted.scaler
    model = fitted.model
    required_dto_version = _required_dto_version_from_metrics(metrics)
    return {
        "model_type": "ridge_linear_combat_value",
        "artifact_schema_version": LEARNED_VALUE_ARTIFACT_SCHEMA_VERSION,
        "feature_schema_version": VALUE_FEATURE_SCHEMA_VERSION,
        "oracle_record_schema_version": ORACLE_RECORD_SCHEMA_VERSION,
        "required_mask_version": ORACLE_VALUE_MASK_VERSION,
        "required_dto_version": required_dto_version,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "training_commit": _training_commit(),
        "feature_schema_hash": hashlib.sha256(
            "\n".join(VALUE_FEATURE_NAMES).encode("utf-8")
        ).hexdigest(),
        "feature_names": list(VALUE_FEATURE_NAMES),
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "terminal_values": terminal_values,
        "training": {
            "objective": "weighted_ridge_regression",
            "alpha": alpha,
            "terminal_weight": terminal_weight,
            "bootstrap_weight": bootstrap_weight,
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
    allow_mixed_teachers: bool,
) -> tuple[list[CombatValueTrainingExample], CombatValueDatasetStats]:
    if not paths:
        return [], CombatValueDatasetStats(0, 0, 0, 0)
    return load_combat_value_examples(
        paths,
        terminal_weight=terminal_weight,
        bootstrap_weight=bootstrap_weight,
        allow_mixed_teachers=allow_mixed_teachers,
    )


def _r2_or_none(targets: list[float], predictions: list[float]) -> float | None:
    if len(targets) < 2:
        return None
    mean = sum(targets) / len(targets)
    total = sum((target - mean) ** 2 for target in targets)
    if total == 0.0:
        return None
    residual = sum(
        (target - prediction) ** 2
        for target, prediction in zip(targets, predictions, strict=True)
    )
    return 1.0 - residual / total


def _terminal_outcome(dto: Any) -> str | None:
    if not isinstance(dto, dict):
        return None
    outcome = dto.get("outcome")
    if outcome in {"victory", "run_victory"}:
        return "victory"
    if outcome == "defeat":
        return "defeat"
    transition = dto.get("transition")
    if isinstance(transition, dict) and transition.get("kind") == "combat_completed":
        if transition.get("victory") is True:
            return "victory"
        if transition.get("victory") is False:
            return "defeat"
    return None


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
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--terminal-weight", type=float, default=1.0)
    parser.add_argument("--bootstrap-weight", type=float, default=0.5)
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
    )
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
    )
    train_examples, train_stats = _load(
        split.train,
        terminal_weight=args.terminal_weight,
        bootstrap_weight=args.bootstrap_weight,
        allow_mixed_teachers=args.allow_mixed_teachers,
    )
    val_examples, val_stats = _load(
        split.val,
        terminal_weight=args.terminal_weight,
        bootstrap_weight=args.bootstrap_weight,
        allow_mixed_teachers=args.allow_mixed_teachers,
    )
    test_examples, test_stats = _load(
        split.test,
        terminal_weight=args.terminal_weight,
        bootstrap_weight=args.bootstrap_weight,
        allow_mixed_teachers=args.allow_mixed_teachers,
    )
    fitted = fit_value_model(train_examples, alpha=args.alpha)
    metrics = {
        "train": evaluate_value_model(fitted, train_examples, train_stats),
        "val": evaluate_value_model(fitted, val_examples, val_stats),
        "test": evaluate_value_model(fitted, test_examples, test_stats),
    }
    payload = weights_payload(
        fitted,
        metrics=metrics,
        split=split,
        dataset_files=paths,
        alpha=args.alpha,
        terminal_weight=args.terminal_weight,
        bootstrap_weight=args.bootstrap_weight,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        training_teacher_provenance=training_provenance.to_json(),
        dataset_teacher_provenance=dataset_provenance.to_json(),
        terminal_values=derive_terminal_values(
            train_examples,
            training_provenance.to_json(),
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
