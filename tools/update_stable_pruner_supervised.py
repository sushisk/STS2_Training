"""Continue supervised stable-pruner learning from an existing artifact.

Unlike the initial sklearn fit, this updater keeps the artifact's feature scale fixed and
applies full-batch weighted logistic-gradient steps to new Oracle v3 pairwise examples.
That makes resume semantics explicit and preserves runtime score compatibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
_REPO_ROOT = _SRC_ROOT.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from sts2_training.decision.learned_pruner import (
    LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION,
    LinearStableFrontierPruner,
)
from sts2_training.decision.oracle_teacher_provenance import (
    inspect_oracle_teacher_provenance,
    require_matching_teacher_provenance,
)
from sts2_training.decision.pruner_features import PRUNER_FEATURE_NAMES
from sts2_training.decision.pruner_training_data import (
    PairwisePrunerExample,
    build_pairwise_examples,
    load_pruner_frontiers,
)


@dataclass(frozen=True)
class SupervisedResumeResult:
    examples: int
    epochs: int
    mean_weighted_log_loss_before: float
    mean_weighted_log_loss_after: float
    gradient_norm_last_epoch: float
    coefficient_delta_norm: float


def apply_supervised_resume(
    coefficients: Sequence[float],
    scale: Sequence[float],
    pairs: Sequence[PairwisePrunerExample],
    *,
    learning_rate: float,
    epochs: int,
    gradient_clip_norm: float,
) -> tuple[list[float], SupervisedResumeResult]:
    _positive_finite("learning_rate", learning_rate)
    _positive_finite("gradient_clip_norm", gradient_clip_norm)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if not pairs:
        raise ValueError("no usable supervised pairwise examples found")
    if len(coefficients) != len(scale):
        raise ValueError("scale must match coefficients")
    if not all(math.isfinite(float(value)) for value in coefficients):
        raise ValueError("coefficients must be finite")
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in scale):
        raise ValueError("scale must contain finite positive numbers")

    current = [float(value) for value in coefficients]
    before = _weighted_log_loss(current, scale, pairs)
    start = tuple(current)
    last_norm = 0.0

    for _ in range(epochs):
        gradient = _weighted_gradient(current, scale, pairs)
        norm = math.sqrt(sum(value * value for value in gradient))
        last_norm = norm
        if norm > float(gradient_clip_norm):
            factor = float(gradient_clip_norm) / norm
            gradient = [value * factor for value in gradient]
        current = [
            coefficient - float(learning_rate) * grad
            for coefficient, grad in zip(current, gradient, strict=True)
        ]
        if not all(math.isfinite(value) for value in current):
            raise ValueError("supervised resume produced non-finite coefficients")

    after = _weighted_log_loss(current, scale, pairs)
    delta_norm = math.sqrt(
        sum((new - old) ** 2 for new, old in zip(current, start, strict=True))
    )
    return current, SupervisedResumeResult(
        examples=len(pairs),
        epochs=epochs,
        mean_weighted_log_loss_before=before,
        mean_weighted_log_loss_after=after,
        gradient_norm_last_epoch=last_norm,
        coefficient_delta_norm=delta_norm,
    )


def updated_artifact_payload(
    base_payload: Mapping[str, Any],
    *,
    base_artifact_sha256: str,
    coefficients: Sequence[float],
    result: SupervisedResumeResult,
    source_files: Sequence[Path],
    teacher_provenance: Mapping[str, Any],
    learning_rate: float,
    epochs: int,
    gradient_clip_norm: float,
    min_target_gap: float,
    terminal_weight: float,
    bootstrap_weight: float,
) -> dict[str, Any]:
    payload = dict(base_payload)
    history_raw = payload.get("supervised_finetuning_history")
    history = list(history_raw) if isinstance(history_raw, list) else []
    update = {
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "training_commit": _training_commit(),
        "parent_artifact_sha256": base_artifact_sha256,
        "source_files": [str(path) for path in source_files],
        "algorithm": "weighted_pairwise_logistic_full_batch_gradient",
        "learning_rate": float(learning_rate),
        "epochs": int(epochs),
        "gradient_clip_norm": float(gradient_clip_norm),
        "min_target_gap": float(min_target_gap),
        "terminal_weight": float(terminal_weight),
        "bootstrap_weight": float(bootstrap_weight),
        "oracle_teacher_provenance": dict(teacher_provenance),
        "result": result.__dict__,
    }
    history.append(update)
    payload["coefficients"] = [float(value) for value in coefficients]
    payload["supervised_finetuning_history"] = history
    payload["last_supervised_update"] = update
    return payload


def resolve_resume_config(
    base_payload: Mapping[str, Any],
    *,
    min_target_gap: float | None,
    terminal_weight: float | None,
    bootstrap_weight: float | None,
) -> tuple[float, float, float]:
    training = base_payload.get("training")
    training = training if isinstance(training, Mapping) else {}
    return (
        _non_negative_finite(
            "min_target_gap",
            training.get("min_target_gap", 1e-6) if min_target_gap is None else min_target_gap,
        ),
        _positive_finite_value(
            "terminal_weight",
            training.get("terminal_weight", 1.0) if terminal_weight is None else terminal_weight,
        ),
        _positive_finite_value(
            "bootstrap_weight",
            training.get("bootstrap_weight", 0.5) if bootstrap_weight is None else bootstrap_weight,
        ),
    )


def _weighted_gradient(
    coefficients: Sequence[float],
    scale: Sequence[float],
    pairs: Sequence[PairwisePrunerExample],
) -> list[float]:
    total_weight = sum(float(pair.weight) for pair in pairs)
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("pair weights must have a finite positive total")
    gradient = [0.0] * len(coefficients)
    for pair in pairs:
        weight = float(pair.weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("pair weights must be finite positive numbers")
        normalized = [
            float(value) / float(factor)
            for value, factor in zip(pair.features, scale, strict=True)
        ]
        score = sum(
            coefficient * value
            for coefficient, value in zip(coefficients, normalized, strict=True)
        )
        probability = _sigmoid(score)
        error = probability - float(pair.label)
        for index, value in enumerate(normalized):
            gradient[index] += weight * error * value
    return [value / total_weight for value in gradient]


def _weighted_log_loss(
    coefficients: Sequence[float],
    scale: Sequence[float],
    pairs: Sequence[PairwisePrunerExample],
) -> float:
    weighted = 0.0
    total_weight = 0.0
    for pair in pairs:
        normalized = [
            float(value) / float(factor)
            for value, factor in zip(pair.features, scale, strict=True)
        ]
        score = sum(
            coefficient * value
            for coefficient, value in zip(coefficients, normalized, strict=True)
        )
        if pair.label == 1:
            loss = _softplus(-score)
        elif pair.label == 0:
            loss = _softplus(score)
        else:
            raise ValueError("pair label must be 0 or 1")
        weight = float(pair.weight)
        weighted += weight * loss
        total_weight += weight
    if total_weight <= 0:
        raise ValueError("pair weights must have a positive total")
    return weighted / total_weight


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def _softplus(value: float) -> float:
    if value > 0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def _positive_finite(name: str, value: float) -> None:
    _positive_finite_value(name, value)


def _positive_finite_value(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _non_negative_finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--min-target-gap", type=float, default=None)
    parser.add_argument("--terminal-weight", type=float, default=None)
    parser.add_argument("--bootstrap-weight", type=float, default=None)
    parser.add_argument("--allow-mixed-teachers", action="store_true")
    parser.add_argument("--allow-teacher-mismatch", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    raw = args.weights.read_bytes()
    base_sha = hashlib.sha256(raw).hexdigest()
    base_payload = json.loads(raw.decode("utf-8"))
    if not isinstance(base_payload, Mapping):
        raise ValueError("learned pruner artifact must be a JSON object")

    LinearStableFrontierPruner.from_weights_file(args.weights)
    if base_payload.get("artifact_schema_version") != LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported learned-pruner artifact schema")
    if tuple(base_payload.get("feature_names") or ()) != PRUNER_FEATURE_NAMES:
        raise ValueError("artifact feature_names do not match runtime schema")

    paths = sorted(args.log_dir.glob("*.jsonl"))
    if not paths:
        raise ValueError(f"no normalized Oracle JSONL found under {args.log_dir}")
    teacher_summary = inspect_oracle_teacher_provenance(
        paths,
        allow_mixed_teachers=args.allow_mixed_teachers,
    )
    require_matching_teacher_provenance(
        base_payload.get("oracle_teacher_provenance"),
        teacher_summary,
        allow_teacher_mismatch=args.allow_teacher_mismatch,
    )
    min_gap, terminal_weight, bootstrap_weight = resolve_resume_config(
        base_payload,
        min_target_gap=args.min_target_gap,
        terminal_weight=args.terminal_weight,
        bootstrap_weight=args.bootstrap_weight,
    )
    frontiers = load_pruner_frontiers(
        paths,
        terminal_weight=terminal_weight,
        bootstrap_weight=bootstrap_weight,
        allow_mixed_teachers=args.allow_mixed_teachers,
    )
    pairs = build_pairwise_examples(frontiers, min_target_gap=min_gap)
    coefficients = [float(value) for value in base_payload.get("coefficients") or ()]
    scale = [float(value) for value in base_payload.get("scale") or ()]
    updated, result = apply_supervised_resume(
        coefficients,
        scale,
        pairs,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        gradient_clip_norm=args.gradient_clip_norm,
    )
    payload = updated_artifact_payload(
        base_payload,
        base_artifact_sha256=base_sha,
        coefficients=updated,
        result=result,
        source_files=paths,
        teacher_provenance=teacher_summary.to_json(),
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        gradient_clip_norm=args.gradient_clip_norm,
        min_target_gap=min_gap,
        terminal_weight=terminal_weight,
        bootstrap_weight=bootstrap_weight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
