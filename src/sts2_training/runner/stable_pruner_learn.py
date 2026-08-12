"""One-line learning/validation entrypoint for stable-pruner logs.

Inputs are log files/directories emitted by the current Training code. The command extracts
current Oracle and RL trajectory records, performs temporary JSONL normalization, and runs
the requested stable-pruner operation.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from sts2_training.decision.oracle_log import ORACLE_RECORD_SCHEMA_VERSION
from sts2_training.decision.pruner_features import PRUNER_FEATURE_SCHEMA_VERSION
from sts2_training.decision.stable_pruner import STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION
from sts2_training.runner.stable_pruner_rl import RL_TRAJECTORY_SCHEMA_VERSION

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRAIN_TOOL = _REPO_ROOT / "tools" / "train_stable_pruner.py"
_EVAL_TOOL = _REPO_ROOT / "tools" / "eval_stable_pruner.py"
_SUPERVISED_RESUME_TOOL = _REPO_ROOT / "tools" / "update_stable_pruner_supervised.py"
_RL_UPDATE_TOOL = _REPO_ROOT / "tools" / "update_stable_pruner_rl.py"
_SUPPORTED_DISCOVERY_SUFFIXES = frozenset({".jsonl", ".json", ".log", ".txt"})
_INGEST_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class SourceLogSummary:
    path: str
    sha256: str
    oracle_records: int
    rl_records: int
    other_json_records: int
    malformed_nonempty_lines: int


@dataclass(frozen=True)
class PreparedLogs:
    oracle_dir: Path
    rl_files: tuple[Path, ...]
    staged_to_source: dict[str, str]
    sources: tuple[SourceLogSummary, ...]
    oracle_records: int
    rl_records: int
    other_json_records: int
    malformed_nonempty_lines: int


@dataclass(frozen=True)
class LearningPlan:
    learn: str
    start: str


@dataclass(frozen=True)
class LearningRunSummary:
    learn: str
    start: str
    output: str
    source_files: int
    oracle_records: int
    rl_records: int
    deferred_records: int
    other_json_records: int
    malformed_nonempty_lines: int
    data_mode: str = "auto-split"
    output_kind: str = "artifact"


def discover_log_files(inputs: Sequence[str]) -> tuple[Path, ...]:
    """Resolve files, directories, and shell-style glob patterns deterministically."""
    if not inputs:
        raise ValueError("at least one log input is required")
    discovered: list[Path] = []
    seen: set[Path] = set()
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.exists():
            if path.is_file():
                matches = [path]
            elif path.is_dir():
                matches = sorted(
                    candidate
                    for candidate in path.rglob("*")
                    if candidate.is_file()
                    and candidate.suffix.lower() in _SUPPORTED_DISCOVERY_SUFFIXES
                )
            else:
                matches = []
        else:
            matches = sorted(Path(value) for value in glob.glob(raw, recursive=True))
            matches = [candidate for candidate in matches if candidate.is_file()]
        if not matches:
            raise ValueError(f"log input did not resolve to any files: {raw}")
        for candidate in matches:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                discovered.append(resolved)
    return tuple(sorted(discovered, key=lambda value: str(value)))


def prepare_logs(inputs: Sequence[str], *, staging_root: Path) -> PreparedLogs:
    """Extract current Oracle/RL records while retaining one staged file per source log."""
    files = discover_log_files(inputs)
    oracle_dir = staging_root / "oracle"
    rl_dir = staging_root / "rl"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    rl_dir.mkdir(parents=True, exist_ok=True)

    staged_to_source: dict[str, str] = {}
    summaries: list[SourceLogSummary] = []
    total_oracle = total_rl = total_other = total_malformed = 0
    rl_files: list[Path] = []

    for index, source in enumerate(files):
        source_display = _display_path(source)
        oracle_records: list[Mapping[str, Any]] = []
        rl_records: list[Mapping[str, Any]] = []
        other_json = malformed = 0
        with source.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = _json_object_from_log_line(line)
                if record is None:
                    malformed += 1
                    continue
                record_type = record.get("record_type")
                if record_type == "combat_oracle_decision":
                    oracle_records.append(record)
                elif record_type == "stable_pruner_rl_episode":
                    rl_records.append(record)
                else:
                    other_json += 1

        prefix = f"{index:05d}-{hashlib.sha256(source_display.encode('utf-8')).hexdigest()[:12]}"
        if oracle_records:
            staged = oracle_dir / f"{prefix}.jsonl"
            _write_records(staged, oracle_records)
            staged_to_source[str(staged)] = source_display
        if rl_records:
            staged = rl_dir / f"{prefix}.jsonl"
            _write_records(staged, rl_records)
            staged_to_source[str(staged)] = source_display
            rl_files.append(staged)

        summary = SourceLogSummary(
            path=source_display,
            sha256=_file_sha256(source),
            oracle_records=len(oracle_records),
            rl_records=len(rl_records),
            other_json_records=other_json,
            malformed_nonempty_lines=malformed,
        )
        summaries.append(summary)
        total_oracle += summary.oracle_records
        total_rl += summary.rl_records
        total_other += summary.other_json_records
        total_malformed += summary.malformed_nonempty_lines

    return PreparedLogs(
        oracle_dir=oracle_dir,
        rl_files=tuple(rl_files),
        staged_to_source=staged_to_source,
        sources=tuple(summaries),
        oracle_records=total_oracle,
        rl_records=total_rl,
        other_json_records=total_other,
        malformed_nonempty_lines=total_malformed,
    )


def resolve_learning_plan(
    requested_learn: str,
    requested_start: str,
    *,
    weights: Path | None,
    oracle_records: int,
    rl_records: int,
) -> LearningPlan:
    if requested_learn not in {"auto", "supervised", "rl"}:
        raise ValueError(f"unsupported learning type: {requested_learn}")
    if requested_start not in {"auto", "fresh", "resume"}:
        raise ValueError(f"unsupported start mode: {requested_start}")

    start = requested_start
    if start == "auto":
        start = "resume" if weights is not None else "fresh"
    if start == "fresh" and weights is not None:
        raise ValueError("fresh start must not receive --weights; remove it or use --start resume")
    if start == "resume" and weights is None:
        raise ValueError("resume requires --weights pointing to the existing artifact")

    learn = requested_learn
    if learn == "auto":
        if start == "fresh":
            learn = "supervised"
        elif oracle_records > 0 and rl_records == 0:
            learn = "supervised"
        elif rl_records > 0 and oracle_records == 0:
            learn = "rl"
        elif oracle_records > 0 and rl_records > 0:
            raise ValueError(
                "both Oracle and RL records are present for resume; specify "
                "--learn supervised or --learn rl"
            )
        else:
            raise ValueError("no learnable stable-pruner records found")

    if learn == "supervised":
        if oracle_records <= 0:
            raise ValueError("supervised learning requires combat_oracle_decision records")
        return LearningPlan(learn=learn, start=start)

    if start == "fresh":
        raise ValueError(
            "RL cannot start fresh from saved trajectories: every RL trajectory is bound "
            "to the behavior artifact SHA. Bootstrap a supervised artifact first, collect "
            "fresh trajectories under it, then use --learn rl --start resume --weights ..."
        )
    if rl_records <= 0:
        raise ValueError("RL learning requires stable_pruner_rl_episode records")
    return LearningPlan(learn=learn, start=start)


def resolve_data_mode(
    requested: str,
    *,
    plan: LearningPlan,
    weights: Path | None,
    oracle_records: int,
) -> str:
    """Resolve data role independently from objective/start semantics.

    Explicit modes are:
    - ``train``: all selected records update the model; no internal validation split.
    - ``validate``: no update; evaluate an existing supervised artifact on all Oracle records.
    - ``auto-split``: fresh supervised fit with source-file train/val/test splitting.

    ``auto`` preserves the one-line defaults: fresh supervised -> auto-split, otherwise train.
    """
    if requested not in {"auto", "train", "validate", "auto-split"}:
        raise ValueError(f"unsupported data mode: {requested}")
    if requested == "auto":
        return "auto-split" if plan.learn == "supervised" and plan.start == "fresh" else "train"
    if requested == "validate":
        if plan.learn != "supervised":
            raise ValueError("--data-mode validate currently supports supervised Oracle evaluation only")
        if weights is None:
            raise ValueError("--data-mode validate requires --weights pointing to the artifact to evaluate")
        if plan.start != "resume":
            raise ValueError("--data-mode validate evaluates an existing artifact and requires resume semantics")
        if oracle_records <= 0:
            raise ValueError("--data-mode validate requires combat_oracle_decision records")
        return requested
    if requested == "auto-split" and not (
        plan.learn == "supervised" and plan.start == "fresh"
    ):
        raise ValueError("--data-mode auto-split is only valid for fresh supervised learning")
    return requested


def resolve_learning_mode(
    requested: str,
    *,
    weights: Path | None,
    oracle_records: int,
    rl_records: int,
) -> str:
    """Backward-compatible helper for the initial one-line API tests/callers."""
    return resolve_learning_plan(
        requested,
        "auto",
        weights=weights,
        oracle_records=oracle_records,
        rl_records=rl_records,
    ).learn


def run_learning(args: argparse.Namespace) -> LearningRunSummary:
    with tempfile.TemporaryDirectory(prefix="sts2-stable-pruner-learn-") as temp_dir:
        prepared = prepare_logs(args.inputs, staging_root=Path(temp_dir))
        requested_data_mode = getattr(args, "data_mode", "auto")
        requested_learn = args.learn
        plan_rl_records = prepared.rl_records
        if requested_data_mode == "validate":
            if requested_learn == "auto":
                requested_learn = "supervised"
            # Validation consumes only Oracle records; unrelated RL lines are deferred.
            plan_rl_records = 0
        plan = resolve_learning_plan(
            requested_learn,
            args.start,
            weights=args.weights,
            oracle_records=prepared.oracle_records,
            rl_records=plan_rl_records,
        )
        data_mode = resolve_data_mode(
            requested_data_mode,
            plan=plan,
            weights=args.weights,
            oracle_records=prepared.oracle_records,
        )
        output = Path(args.output or _default_output(plan, data_mode=data_mode))
        output_resolved = output.resolve()
        if args.weights is not None and output_resolved == args.weights.resolve():
            raise ValueError(
                "--output must differ from --weights so the parent artifact remains immutable "
                "and its pre-update SHA-256 provenance stays valid"
            )
        source_paths = {
            Path(summary.path).expanduser().resolve()
            for summary in prepared.sources
        }
        if output_resolved in source_paths:
            raise ValueError(
                "--output must differ from every resolved input log so source training data "
                "cannot be overwritten by an artifact or validation report"
            )
        output.parent.mkdir(parents=True, exist_ok=True)

        if data_mode == "validate":
            command = _supervised_validation_command(args, prepared=prepared)
            _run_validation_command(command, output=output)
            deferred = prepared.rl_records
            output_kind = "validation_report"
        elif plan.learn == "supervised" and plan.start == "fresh":
            command = _supervised_fresh_command(
                args, prepared=prepared, output=output, data_mode=data_mode
            )
            _run_command(command)
            deferred = prepared.rl_records
            output_kind = "artifact"
        elif plan.learn == "supervised":
            command = _supervised_resume_command(args, prepared=prepared, output=output)
            _run_command(command)
            deferred = prepared.rl_records
            output_kind = "artifact"
        else:
            if data_mode != "train":
                raise ValueError("RL trajectory updates require --data-mode train")
            command = _rl_resume_command(args, prepared=prepared, output=output)
            _run_command(command)
            deferred = prepared.oracle_records
            output_kind = "artifact"

        if not output.exists():
            raise RuntimeError(f"stable-pruner command completed without creating output: {output}")
        _rewrite_output_provenance(
            output,
            plan=plan,
            data_mode=data_mode,
            output_kind=output_kind,
            prepared=prepared,
            base_weights=args.weights,
        )
        return LearningRunSummary(
            learn=plan.learn,
            start=plan.start,
            output=str(output),
            source_files=len(prepared.sources),
            oracle_records=prepared.oracle_records,
            rl_records=prepared.rl_records,
            deferred_records=deferred,
            other_json_records=prepared.other_json_records,
            malformed_nonempty_lines=prepared.malformed_nonempty_lines,
            data_mode=data_mode,
            output_kind=output_kind,
        )


def _supervised_fresh_command(
    args: argparse.Namespace,
    *,
    prepared: PreparedLogs,
    output: Path,
    data_mode: str = "auto-split",
) -> list[str]:
    _require_tool(_TRAIN_TOOL)
    if data_mode not in {"auto-split", "train"}:
        raise ValueError("fresh supervised command requires auto-split or train data mode")
    min_target_gap = 1e-6 if args.min_target_gap is None else args.min_target_gap
    terminal_weight = 1.0 if args.terminal_weight is None else args.terminal_weight
    bootstrap_weight = 0.5 if args.bootstrap_weight is None else args.bootstrap_weight
    val_fraction = 0.0 if data_mode == "train" else args.val_fraction
    test_fraction = 0.0 if data_mode == "train" else args.test_fraction
    command = [
        sys.executable,
        str(_TRAIN_TOOL),
        "--log-dir",
        str(prepared.oracle_dir),
        "--output",
        str(output),
        "--val-fraction",
        str(val_fraction),
        "--test-fraction",
        str(test_fraction),
        "--seed",
        str(args.seed),
        "--inverse-regularization",
        str(args.inverse_regularization),
        "--min-target-gap",
        str(min_target_gap),
        "--terminal-weight",
        str(terminal_weight),
        "--bootstrap-weight",
        str(bootstrap_weight),
    ]
    if args.allow_mixed_teachers:
        command.append("--allow-mixed-teachers")
    return command


def _supervised_validation_command(
    args: argparse.Namespace, *, prepared: PreparedLogs
) -> list[str]:
    _require_tool(_EVAL_TOOL)
    if args.weights is None:
        raise ValueError("validation requires --weights")
    command = [
        sys.executable,
        str(_EVAL_TOOL),
        "--weights",
        str(args.weights),
        "--log-dir",
        str(prepared.oracle_dir),
    ]
    if args.min_target_gap is not None:
        command.extend(["--min-target-gap", str(args.min_target_gap)])
    if args.terminal_weight is not None:
        command.extend(["--terminal-weight", str(args.terminal_weight)])
    if args.bootstrap_weight is not None:
        command.extend(["--bootstrap-weight", str(args.bootstrap_weight)])
    if args.allow_mixed_teachers:
        command.append("--allow-mixed-teachers")
    if args.allow_teacher_mismatch:
        command.append("--allow-teacher-mismatch")
    return command


def _supervised_resume_command(
    args: argparse.Namespace,
    *,
    prepared: PreparedLogs,
    output: Path,
) -> list[str]:
    _require_tool(_SUPERVISED_RESUME_TOOL)
    assert args.weights is not None
    command = [
        sys.executable,
        str(_SUPERVISED_RESUME_TOOL),
        "--weights",
        str(args.weights),
        "--log-dir",
        str(prepared.oracle_dir),
        "--output",
        str(output),
        "--learning-rate",
        str(args.learning_rate),
        "--epochs",
        str(args.epochs),
        "--gradient-clip-norm",
        str(args.gradient_clip_norm),
    ]
    if args.min_target_gap is not None:
        command.extend(["--min-target-gap", str(args.min_target_gap)])
    if args.terminal_weight is not None:
        command.extend(["--terminal-weight", str(args.terminal_weight)])
    if args.bootstrap_weight is not None:
        command.extend(["--bootstrap-weight", str(args.bootstrap_weight)])
    if args.allow_mixed_teachers:
        command.append("--allow-mixed-teachers")
    if args.allow_teacher_mismatch:
        command.append("--allow-teacher-mismatch")
    return command


def _rl_resume_command(
    args: argparse.Namespace,
    *,
    prepared: PreparedLogs,
    output: Path,
) -> list[str]:
    _require_tool(_RL_UPDATE_TOOL)
    assert args.weights is not None
    command = [
        sys.executable,
        str(_RL_UPDATE_TOOL),
        "--weights",
        str(args.weights),
        "--output",
        str(output),
        "--learning-rate",
        str(args.learning_rate),
        "--gradient-clip-norm",
        str(args.gradient_clip_norm),
    ]
    for path in prepared.rl_files:
        command.extend(["--trajectory", str(path)])
    return command


def _run_command(command: Sequence[str]) -> None:
    try:
        subprocess.run(list(command), cwd=_REPO_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"learning command failed with exit status {exc.returncode}: " + " ".join(command)
        ) from exc


def _run_validation_command(command: Sequence[str], *, output: Path) -> None:
    try:
        completed = subprocess.run(
            list(command),
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(
            f"validation command failed with exit status {exc.returncode}: "
            + " ".join(command)
            + suffix
        ) from exc
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("validation command did not emit a JSON report") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("validation command JSON report must be an object")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _rewrite_output_provenance(
    output: Path,
    *,
    plan: LearningPlan,
    data_mode: str,
    output_kind: str,
    prepared: PreparedLogs,
    base_weights: Path | None,
) -> None:
    raw = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("stable-pruner output must be a JSON object")
    payload = _replace_staged_paths(dict(raw), prepared.staged_to_source)
    payload["one_line_learning_ingest"] = {
        "schema_version": _INGEST_SCHEMA_VERSION,
        "learn": plan.learn,
        "start": plan.start,
        "data_mode": data_mode,
        "output_kind": output_kind,
        "updates_model": output_kind == "artifact",
        "oracle_record_schema_version": ORACLE_RECORD_SCHEMA_VERSION,
        "rl_trajectory_schema_version": RL_TRAJECTORY_SCHEMA_VERSION,
        "stable_prune_node_view_schema_version": STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
        "feature_schema_version": PRUNER_FEATURE_SCHEMA_VERSION,
        "base_artifact_sha256": None if base_weights is None else _file_sha256(base_weights),
        "source_logs": [asdict(summary) for summary in prepared.sources],
        "record_counts": {
            "combat_oracle_decision": prepared.oracle_records,
            "stable_pruner_rl_episode": prepared.rl_records,
            "other_json": prepared.other_json_records,
            "malformed_nonempty_lines": prepared.malformed_nonempty_lines,
        },
        "normalization": "extract-json-object-per-line-v1",
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def _rewrite_artifact_provenance(
    output: Path,
    *,
    plan: LearningPlan,
    prepared: PreparedLogs,
    base_weights: Path | None,
) -> None:
    """Backward-compatible helper retained for tests/callers created before data modes."""
    _rewrite_output_provenance(
        output,
        plan=plan,
        data_mode="auto-split" if plan.start == "fresh" else "train",
        output_kind="artifact",
        prepared=prepared,
        base_weights=base_weights,
    )


def _replace_staged_paths(value: Any, mapping: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, Mapping):
        return {str(key): _replace_staged_paths(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_staged_paths(item, mapping) for item in value]
    return value


def _json_object_from_log_line(line: str) -> Mapping[str, Any] | None:
    text = line.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, Mapping) else None


def _write_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            handle.write("\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(resolved.relative_to(cwd))
    except ValueError:
        return str(resolved)


def _default_output(plan: LearningPlan, *, data_mode: str = "auto-split") -> Path:
    if data_mode == "validate":
        return Path("tools/output/stable_pruner_validation.json")
    if plan.learn == "supervised" and plan.start == "fresh":
        return Path("tools/output/stable_pruner_weights.json")
    if plan.learn == "supervised":
        return Path("tools/output/stable_pruner_supervised_resumed_weights.json")
    return Path("tools/output/stable_pruner_rl_weights.json")


def _require_tool(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(
            f"required training tool is unavailable at {path}; run this command from a "
            "source checkout of STS2_Training"
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="log file(s), directories, or glob patterns; directories are searched recursively",
    )
    parser.add_argument(
        "--learn",
        "--mode",
        dest="learn",
        choices=("auto", "supervised", "rl"),
        default="auto",
        help="what objective to optimize; auto infers from start mode and record types",
    )
    parser.add_argument(
        "--start",
        choices=("auto", "fresh", "resume"),
        default="auto",
        help="fresh starts from zero; resume requires --weights; auto uses --weights presence",
    )
    parser.add_argument(
        "--data-mode",
        choices=("auto", "train", "validate", "auto-split"),
        default="auto",
        help=(
            "data role: train uses all selected data for updates; validate performs no update; "
            "auto-split performs fresh supervised train/val/test splitting; auto preserves "
            "one-line defaults"
        ),
    )
    parser.add_argument(
        "--weights", type=Path, default=None, help="existing artifact for resume or validation"
    )
    parser.add_argument("--output", type=Path, default=None)

    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inverse-regularization", type=float, default=1.0)
    parser.add_argument("--min-target-gap", type=float, default=None)
    parser.add_argument("--terminal-weight", type=float, default=None)
    parser.add_argument("--bootstrap-weight", type=float, default=None)
    parser.add_argument("--allow-mixed-teachers", action="store_true")
    parser.add_argument("--allow-teacher-mismatch", action="store_true")

    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        summary = run_learning(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"stable-pruner operation failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True))
    if summary.deferred_records:
        deferred_type = (
            "stable_pruner_rl_episode" if summary.learn == "supervised" else "combat_oracle_decision"
        )
        print(
            f"note: {summary.deferred_records} {deferred_type} records were detected but are "
            "not part of the selected stable-pruner operation",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
