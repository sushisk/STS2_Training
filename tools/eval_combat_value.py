"""Evaluate a learned Combat ValueModel on held-out Oracle v5 JSONL."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from sts2_training.decision.learned_value import LinearValueModel
from sts2_training.decision.oracle_teacher_provenance import (
    inspect_oracle_teacher_provenance,
    require_matching_teacher_provenance,
)
from sts2_training.decision.value_training_data import load_combat_value_examples


def _metrics(model: LinearValueModel, examples, stats):
    result = stats.to_json()
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
    predictions = [model.evaluate(example.masked_emulator_dto) for example in examples]
    errors = [
        prediction - example.target_value
        for prediction, example in zip(predictions, examples, strict=True)
    ]
    weights = [example.sample_weight for example in examples]
    weight_total = sum(weights)
    targets = [example.target_value for example in examples]
    mean = sum(targets) / len(targets)
    total = sum((target - mean) ** 2 for target in targets)
    residual = sum(
        (target - prediction) ** 2
        for target, prediction in zip(targets, predictions, strict=True)
    )
    result.update(
        {
            "mae": sum(abs(error) for error in errors) / len(errors),
            "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
            "weighted_mae": sum(
                weight * abs(error) for weight, error in zip(weights, errors, strict=True)
            )
            / weight_total,
            "weighted_rmse": math.sqrt(
                sum(
                    weight * error * error
                    for weight, error in zip(weights, errors, strict=True)
                )
                / weight_total
            ),
            "r2": None if len(targets) < 2 or total == 0.0 else 1.0 - residual / total,
        }
    )
    return result


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--terminal-weight", type=float, default=1.0)
    parser.add_argument("--bootstrap-weight", type=float, default=0.5)
    parser.add_argument("--allow-mixed-teachers", action="store_true")
    parser.add_argument("--allow-teacher-mismatch", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    paths = sorted(args.log_dir.glob("*.jsonl"))
    if not paths:
        print(f"no *.jsonl files found under {args.log_dir}", file=sys.stderr)
        return 1
    payload = json.loads(args.weights.read_text(encoding="utf-8"))
    evaluation_provenance = inspect_oracle_teacher_provenance(
        paths,
        allow_mixed_teachers=args.allow_mixed_teachers,
    )
    teacher_matches = require_matching_teacher_provenance(
        payload.get("oracle_teacher_provenance"),
        evaluation_provenance,
        allow_teacher_mismatch=args.allow_teacher_mismatch,
    )
    examples, stats = load_combat_value_examples(
        paths,
        terminal_weight=args.terminal_weight,
        bootstrap_weight=args.bootstrap_weight,
        allow_mixed_teachers=args.allow_mixed_teachers,
    )
    model = LinearValueModel.from_weights_file(args.weights)
    report = {
        "teacher_matches_artifact": teacher_matches,
        "oracle_teacher_provenance": evaluation_provenance.to_json(),
        "metrics": _metrics(model, examples, stats),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
