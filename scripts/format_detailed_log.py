#!/usr/bin/env python3
"""Format a detailed Whole Run JSONL log for human and AI inspection.

The source JSONL is preserved.  By default this writes a compact Markdown report
and a normalized JSON summary to the current working directory.  For an input named
``run.jsonl``, the default outputs are ``tmp_analyze_run.md`` and
``tmp_analyze_run.json``.  Use ``--all-roots`` for a complete decision timeline.

Examples::

    python scripts/format_detailed_log.py data/evaluation/detailed_logs/run.jsonl
    python scripts/format_detailed_log.py run.jsonl --all-roots
    python scripts/format_detailed_log.py run.jsonl --output custom.md --json-output custom.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"{path}:{line_number}: event must be a JSON object")
            events.append(event)
    return events


def _number(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value) if value is not None else "-"


def _board(dto: Mapping[str, Any]) -> dict[str, Any]:
    """Select stable, high-value board fields without discarding the raw DTO."""
    fields = (
        "boundary", "totalFloor", "actFloor", "currentActIndex", "currentRoomType",
        "stepIndex", "hp", "maxHp", "block", "energy", "maxEnergy", "gold",
        "choiceScope", "isGameOver",
    )
    board = {key: dto[key] for key in fields if key in dto}
    for key in ("enemies", "enemyStates", "playerPowers", "relics", "potions", "legal_actions"):
        if key in dto:
            board[key] = dto[key]
    return board


def _root_record(event: Mapping[str, Any]) -> dict[str, Any]:
    dto = event.get("masked_emulator_dto")
    dto = dto if isinstance(dto, Mapping) else {}
    scores = event.get("action_scores")
    scores = scores if isinstance(scores, list) else []
    return {
        "decision_point_id": event.get("decision_point_id"),
        "instance_id": event.get("instance_id"),
        "decision_source": event.get("decision_source"),
        "selected_action_id": event.get("selected_action_id"),
        "board": _board(dto),
        "action_scores": scores,
    }


def _action_label(root: Mapping[str, Any]) -> str:
    selected = root.get("selected_action_id")
    for action in root.get("action_scores", []):
        if isinstance(action, Mapping) and action.get("action_id") == selected:
            return str(action.get("label", selected))
    return str(selected) if selected is not None else "-"


def _markdown(
    path: Path,
    events: list[dict[str, Any]],
    roots: list[dict[str, Any]],
    result: Mapping[str, Any] | None,
    *,
    total_root_count: int,
    all_floors: Iterable[Any],
) -> str:
    counts = Counter(str(event.get("event", "unknown")) for event in events)
    lines = [
        f"# Detailed log: `{path.name}`",
        "",
        "## Summary",
        "",
        f"- Events: {len(events)}",
        f"- Root decisions: {total_root_count}",
        f"- Floors observed: {', '.join(dict.fromkeys(map(str, all_floors))) or '-'}",
        f"- Outcome: {result.get('outcome', '-')}" if result else "- Outcome: not completed",
        f"- Decisions made: {result.get('decisions_made', '-')}" if result else "- Decisions made: -",
        "",
        "Event counts: " + ", ".join(f"`{key}`={value}" for key, value in sorted(counts.items())),
        "",
        "## Root timeline",
        "",
        "| # | Floor | Boundary | HP | Block | Energy | Room | Selected action | Scores |",
        "|---:|---:|---|---:|---:|---:|---|---|---|",
    ]
    for index, root in enumerate(roots, 1):
        board = root["board"]
        hp = f"{_number(board.get('hp'))}/{_number(board.get('maxHp'))}"
        scores = root.get("action_scores", [])
        score_text = ", ".join(
            f"{item.get('label', item.get('action_id'))}={item.get('score')}"
            for item in scores[:5]
            if isinstance(item, Mapping)
        ) or "-"
        lines.append(
            f"| {index} | {_number(board.get('totalFloor'))} | {board.get('boundary', '-')} | "
            f"{hp} | {_number(board.get('block'))} | {_number(board.get('energy'))}/"
            f"{_number(board.get('maxEnergy'))} | {board.get('currentRoomType', '-')} | "
            f"`{_action_label(root)}` | {score_text} |"
        )
    lines.extend(["", "## Selected root details", ""])
    for index, root in enumerate(roots, 1):
        lines.extend(
            [
                f"### Root {index}: `{root.get('decision_point_id')}`",
                "",
                "```json",
                json.dumps(root, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="detailed JSONL log")
    parser.add_argument(
        "--output",
        type=Path,
        help="Markdown output (default: cwd/tmp_analyze_<input_name>.md)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="JSON output (default: cwd/tmp_analyze_<input_name>.json)",
    )
    parser.add_argument("--last-roots", type=int, default=20, help="number of latest roots (default: 20)")
    parser.add_argument("--all-roots", action="store_true", help="include every root in the report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.last_roots < 1:
        raise SystemExit("--last-roots must be positive")
    events = _read_events(args.log)
    all_roots = [_root_record(event) for event in events if event.get("event") == "root_decision"]
    roots = all_roots if args.all_roots else all_roots[-args.last_roots :]
    result_events = [event for event in events if event.get("event") == "run_result"]
    result = result_events[-1] if result_events else None

    all_floors = [
        root["board"].get("totalFloor")
        for root in all_roots
        if root["board"].get("totalFloor") is not None
    ]
    markdown = _markdown(
        args.log,
        events,
        roots,
        result,
        total_root_count=len(all_roots),
        all_floors=all_floors,
    )
    input_name = args.log.stem
    markdown_path = args.output or (Path.cwd() / f"tmp_analyze_{input_name}.md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")

    payload = {
        "source": str(args.log),
        "event_counts": dict(Counter(str(event.get("event", "unknown")) for event in events)),
        "run_result": result,
        "root_decisions": roots,
    }
    json_path = args.json_output or (Path.cwd() / f"tmp_analyze_{input_name}.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
