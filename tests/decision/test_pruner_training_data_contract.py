from __future__ import annotations

import json
from pathlib import Path

import pytest

import sts2_training.decision.pruner_training_data as training_data


def _trace_node(node_id: str, *, index: int) -> dict[str, object]:
    return {
        "node_id": node_id,
        "parent_node_id": "search:root",
        "branch_id": node_id.rsplit(":", 1)[-1],
        "parent_branch_id": "root",
        "frontier_index_before_prune": index,
        "kept": index == 0,
        "value": 10.0 - index,
        "root_action_id": "A",
        "rng_id": 1,
        "decision_point_id": f"d-{index}",
        "depth": 1,
        "combat_depth": 1,
        "continuation_steps": 0,
        "terminal": False,
        "action_id": f"action-{index}",
        "action_type": "card",
        "action": None,
        "policy_rank": index,
        "policy_score": None,
        "post_coverage_rank": index,
        "candidate_source": "policy",
    }


def _stable_prune_event(node_ids: tuple[str, ...]) -> dict[str, object]:
    return {
        "event_type": "stable_prune",
        "search_id": "search",
        "prune_step_id": "prune",
        "phase": "stable_frontier",
        "k": 1,
        "frontier_size": len(node_ids),
        "pruner_name": "value_top_k",
        "pruner_version": "1",
        "max_depth": 3,
        "depths_completed": 0,
        "remaining_time_ms": None,
        "nodes": [
            _trace_node(node_id, index=index)
            for index, node_id in enumerate(node_ids)
        ],
    }


def _target(
    node_id: str,
    *,
    source: str = "value_bootstrap",
    value: float | None = 5.0,
    censored: bool = True,
    censor_reason: str | None = "value_bootstrap:max_depth",
    prune_step_id: str = "prune",
) -> dict[str, object]:
    return {
        "prune_step_id": prune_step_id,
        "node_id": node_id,
        "target_beam_width": 1,
        "target_value": value,
        "target_source": source,
        "baseline_would_keep": True,
        "oracle_kept": True,
        "censored": censored,
        "censor_reason": censor_reason,
    }


def _record(
    *,
    node_ids: tuple[str, ...],
    targets: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "record_type": "combat_oracle_decision",
        "decision_point_id": "decision",
        "search_trace": [_stable_prune_event(node_ids)],
        "oracle_targets": {"stable_nodes": targets},
    }


def _load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record: dict[str, object],
):
    monkeypatch.setattr(
        training_data,
        "inspect_oracle_teacher_provenance",
        lambda *_args, **_kwargs: None,
    )
    path = tmp_path / "oracle.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return training_data.load_pruner_frontiers([path])


def test_missing_one_stable_target_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record(
        node_ids=("search:a", "search:b"),
        targets=[_target("search:a")],
    )

    with pytest.raises(ValueError, match="stable prune target mismatch"):
        _load(tmp_path, monkeypatch, record)


def test_missing_entire_stable_target_step_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record(node_ids=("search:a",), targets=[])

    with pytest.raises(ValueError, match="stable prune target mismatch"):
        _load(tmp_path, monkeypatch, record)


def test_extra_stable_target_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record(
        node_ids=("search:a",),
        targets=[_target("search:a"), _target("search:b")],
    )

    with pytest.raises(ValueError, match="stable prune target mismatch"):
        _load(tmp_path, monkeypatch, record)


def test_explicit_expansion_censoring_from_oracle_is_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record(
        node_ids=("search:a",),
        targets=[
            _target(
                "search:a",
                source="no_target",
                value=None,
                censored=True,
                censor_reason="expansion_attempted_without_outcome:test",
            )
        ],
    )

    frontiers = _load(tmp_path, monkeypatch, record)

    assert len(frontiers) == 1
    assert len(frontiers[0].nodes) == 1
    node = frontiers[0].nodes[0]
    assert node.target_value is None
    assert node.target_source == "no_target"
    assert node.censored is True
    assert node.censor_reason == "expansion_attempted_without_outcome:test"
