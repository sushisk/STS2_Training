"""Check Whole Run score logs for Beam candidates that never reached the frontier.

Acceptance check for the Beam-scope fix: a correctly scoped run drops nothing, so every
root candidate proposed by the policy must appear in the depth-1 stable frontier (or be
accounted for as a branch fault / out-of-scope drop).

Usage::

    python tools/check_score_log_scope.py data/evaluation/score_logs/<dir>

Exits non-zero when any candidate went missing, so it can gate a re-evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _trace_events(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    by_search: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("event") != "score_trace":
                    continue
                event = record.get("trace_event")
                if isinstance(event, dict) and isinstance(event.get("search_id"), str):
                    by_search[event["search_id"]].append(event)
    return by_search


def _report(by_search: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    searches = 0
    searches_with_missing = 0
    missing_by_label: Counter[str] = Counter()
    present_by_label: Counter[str] = Counter()
    targeted_wipeouts = 0
    out_of_scope = 0
    faulted = 0

    for events in by_search.values():
        root_proposal = next(
            (
                event
                for event in events
                if event.get("event_type") == "policy_proposal"
                and event.get("parent_branch_id") == "root"
            ),
            None,
        )
        if root_proposal is None:
            continue
        searches += 1

        first_prune = next(
            (
                event
                for event in events
                if event.get("event_type") == "stable_prune"
                and event.get("depths_completed") == 0
            ),
            None,
        )
        reached = {
            node.get("root_action_id")
            for node in (first_prune.get("nodes", []) if first_prune else [])
        }
        continued = {
            event.get("parent_branch_id")
            for event in events
            if event.get("event_type") == "policy_proposal"
        }

        missing_here = 0
        targeted_total = 0
        targeted_missing = 0
        for candidate in root_proposal.get("candidates", []):
            action = candidate.get("action") or {}
            label = str(action.get("label", "?"))
            targeted = (action.get("parameters") or {}).get("targetType") == "AnyEnemy"
            targeted_total += int(targeted)
            # A candidate is accounted for if it reached the stable frontier or was still
            # being expanded as a continuation branch.
            if candidate.get("action_id") in reached or candidate.get("branch_id") in continued:
                present_by_label[label] += 1
                continue
            missing_by_label[label] += 1
            missing_here += 1
            targeted_missing += int(targeted)

        if missing_here:
            searches_with_missing += 1
        if targeted_total and targeted_missing == targeted_total:
            targeted_wipeouts += 1

        for event in events:
            if event.get("event_type") == "out_of_scope_drop":
                out_of_scope += 1
            elif event.get("event_type") == "branch_fault":
                faulted += 1

    return {
        "searches": searches,
        "searches_with_missing_candidates": searches_with_missing,
        "searches_with_all_targeted_candidates_missing": targeted_wipeouts,
        "out_of_scope_drops": out_of_scope,
        "branch_faults": faulted,
        "missing_by_label": dict(missing_by_label.most_common()),
        "present_by_label": dict(present_by_label.most_common()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="score-log JSONL files, or directories containing them",
    )
    args = parser.parse_args(argv)

    files: list[Path] = []
    for path in args.paths:
        files.extend(sorted(path.rglob("*.jsonl")) if path.is_dir() else [path])
    if not files:
        parser.error("no score-log JSONL files found")

    report = _report(_trace_events(files))
    report["files"] = len(files)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if report["searches"] == 0:
        print("no root proposals found; nothing to check", file=sys.stderr)
        return 1
    return 1 if report["searches_with_missing_candidates"] or report["out_of_scope_drops"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
