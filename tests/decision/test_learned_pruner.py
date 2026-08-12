from __future__ import annotations

import json
from pathlib import Path

import pytest

from sts2_training.decision.engine import CombatDecisionEngine
from sts2_training.decision.learned_pruner import (
    LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION,
    LinearStableFrontierPruner,
)
from sts2_training.decision.oracle_log import ORACLE_RECORD_SCHEMA_VERSION
from sts2_training.decision.pruner_features import (
    PRUNER_FEATURE_NAMES,
    PRUNER_FEATURE_SCHEMA_VERSION,
    stable_pruner_feature_matrix,
)
from sts2_training.decision.pruner_training_data import (
    build_pairwise_examples,
    load_pruner_frontiers,
)
from sts2_training.decision.search_trace import StablePruneNodeTrace, StablePruneTrace
from sts2_training.decision.stable_pruner import (
    STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
    StablePruneContext,
    StablePruneNodeView,
)


def _context() -> StablePruneContext:
    return StablePruneContext(
        search_id="s",
        prune_step_id="s:prune:0",
        phase="stable_frontier",
        beam_width=2,
        max_depth=4,
        depths_completed=1,
        remaining_time_ms=250.0,
    )


def _node(
    value: float,
    *,
    root_action_id: str,
    action_type: str = "card",
    policy_rank: int | None = 0,
) -> StablePruneNodeView:
    return StablePruneNodeView(
        value=value,
        root_action_id=root_action_id,
        depth=1,
        combat_depth=1,
        continuation_steps=0,
        terminal=False,
        action_type=action_type,
        policy_rank=policy_rank,
        policy_score=None,
        post_coverage_rank=policy_rank,
        candidate_source="policy",
    )


def _zero_pruner() -> LinearStableFrontierPruner:
    return LinearStableFrontierPruner(
        feature_names=PRUNER_FEATURE_NAMES,
        coefficients=[0.0] * len(PRUNER_FEATURE_NAMES),
    )


def test_feature_matrix_contains_frontier_and_root_group_context() -> None:
    frontier = [
        _node(5.0, root_action_id="r1"),
        _node(3.0, root_action_id="r1"),
        _node(1.0, root_action_id="r2", action_type="potion"),
    ]

    rows = stable_pruner_feature_matrix(frontier, context=_context())
    by_name = [dict(zip(PRUNER_FEATURE_NAMES, row, strict=True)) for row in rows]

    assert len(rows) == 3
    assert by_name[0]["frontier_value_max"] == 5.0
    assert by_name[0]["root_group_size"] == 2.0
    assert by_name[0]["within_root_value_rank_fraction"] == 0.0
    assert by_name[1]["within_root_value_rank_fraction"] == 1.0
    assert by_name[2]["root_group_size"] == 1.0
    assert by_name[2]["action_type_potion"] == 1.0
    assert by_name[0]["beam_width"] == 2.0
    assert "remaining_depth" not in PRUNER_FEATURE_NAMES
    assert "remaining_time_ms_missing" not in PRUNER_FEATURE_NAMES
    assert "remaining_time_ms_log1p" not in PRUNER_FEATURE_NAMES


def test_linear_pruner_returns_frontier_indices_and_preserves_score_ties() -> None:
    coefficients = [0.0] * len(PRUNER_FEATURE_NAMES)
    coefficients[PRUNER_FEATURE_NAMES.index("action_type_potion")] = 10.0
    pruner = LinearStableFrontierPruner(
        feature_names=PRUNER_FEATURE_NAMES,
        coefficients=coefficients,
    )
    first = _node(10.0, root_action_id="r1")
    potion = _node(1.0, root_action_id="r2", action_type="potion")
    second = _node(9.0, root_action_id="r3")

    selected = pruner.select([first, potion, second], k=2, context=_context())
    assert selected == [1, 0]

    tied = _zero_pruner().select([first, potion, second], k=2, context=_context())
    assert tied == [0, 1]


def test_combat_engine_accepts_learned_stable_pruner_without_widening_its_scope() -> None:
    pruner = _zero_pruner()

    engine = CombatDecisionEngine(object(), stable_pruner=pruner)

    assert engine.beam_search.stable_pruner is pruner


def _artifact_payload() -> dict:
    return {
        "model_type": "pairwise_logistic_linear_pruner",
        "artifact_schema_version": LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION,
        "stable_prune_node_view_schema_version": STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
        "feature_schema_version": PRUNER_FEATURE_SCHEMA_VERSION,
        "feature_names": list(PRUNER_FEATURE_NAMES),
        "coefficients": [0.0] * len(PRUNER_FEATURE_NAMES),
        "scale": [1.0] * len(PRUNER_FEATURE_NAMES),
    }


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("artifact_schema_version", 1, "artifact_schema_version"),
        ("stable_prune_node_view_schema_version", 999, "stable_prune_node_view_schema_version"),
        ("feature_schema_version", 999, "feature_schema_version"),
    ],
)
def test_weights_file_rejects_schema_mismatch(
    tmp_path: Path,
    field: str,
    value: int,
    match: str,
) -> None:
    payload = _artifact_payload()
    payload[field] = value
    path = tmp_path / "weights.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        LinearStableFrontierPruner.from_weights_file(path)


def test_weights_file_rejects_feature_name_mismatch(tmp_path: Path) -> None:
    payload = _artifact_payload()
    payload["feature_names"] = ["wrong"]
    payload["coefficients"] = [1.0]
    payload["scale"] = [1.0]
    path = tmp_path / "weights.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="feature_names"):
        LinearStableFrontierPruner.from_weights_file(path)


def test_runtime_and_trace_replay_views_have_identical_features() -> None:
    runtime_views = (
        _node(8.0, root_action_id="r1", policy_rank=0),
        _node(7.0, root_action_id="r2", policy_rank=1),
        _node(6.0, root_action_id="r3", policy_rank=2),
    )
    trace = StablePruneTrace(
        search_id="s",
        prune_step_id="s:prune:0",
        phase="stable_frontier",
        k=3,
        frontier_size=3,
        pruner_name="value_top_k",
        pruner_version="1",
        max_depth=4,
        depths_completed=1,
        remaining_time_ms=250.0,
        nodes=tuple(
            StablePruneNodeTrace(
                **_trace_node(
                    f"n{index}",
                    view.value,
                    root_action_id=view.root_action_id or "r",
                    index=index,
                )
            )
            for index, view in enumerate(runtime_views)
        ),
    )

    replay_views = trace.node_views()
    replay_context = trace.to_prune_context(beam_width=2)

    assert replay_views == runtime_views
    assert replay_context == _context()
    assert stable_pruner_feature_matrix(replay_views, context=replay_context) == (
        stable_pruner_feature_matrix(runtime_views, context=_context())
    )


def test_oracle_jsonl_keeps_no_target_nodes_but_pairwise_training_excludes_them(
    tmp_path: Path,
) -> None:
    path = tmp_path / "oracle.jsonl"
    nodes = [
        _trace_node("n1", 8.0, root_action_id="r1", index=0),
        _trace_node("n2", 7.0, root_action_id="r2", index=1),
        _trace_node("n3", 6.0, root_action_id="r3", index=2),
    ]
    record = {
        "record_type": "combat_oracle_decision",
        "record_schema_version": ORACLE_RECORD_SCHEMA_VERSION,
        "decision_point_id": "d0",
        "provenance": {
            "training_commit": "test",
            "teacher_policy_class": "test.Policy",
            "teacher_inner_policy_class": "test.Policy",
            "teacher_coverage_policy_class": None,
            "teacher_value_class": "test.Value",
            "teacher_policy_metadata": {},
            "teacher_inner_policy_metadata": {},
            "teacher_value_metadata": {},
            "pruner_name": "value_top_k",
            "pruner_version": "1",
            "rng_sampling": "independent",
        },
        "search_trace": [
            {
                "event_type": "stable_prune",
                "search_id": "s",
                "prune_step_id": "s:prune:0",
                "phase": "stable_frontier",
                "k": 3,
                "frontier_size": 3,
                "pruner_name": "value_top_k",
                "pruner_version": "1",
                "max_depth": 4,
                "depths_completed": 1,
                "remaining_time_ms": 100.0,
                "nodes": nodes,
                "selected_indices": [0, 1],
            }
        ],
        "oracle_targets": {
            "stable_nodes": [
                _target("n1", 20.0, source="terminal", baseline=True, oracle=True),
                _target(
                    "n2",
                    30.0,
                    source="value_bootstrap",
                    baseline=True,
                    oracle=True,
                ),
                {
                    **_target("n3", 0.0, source="no_target", baseline=False, oracle=False),
                    "target_value": None,
                    "censored": True,
                    "censor_reason": "oracle_pruned_before_followup",
                },
            ]
        },
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    frontiers = load_pruner_frontiers([path], terminal_weight=1.0, bootstrap_weight=0.4)
    pairs = build_pairwise_examples(frontiers)

    assert len(frontiers) == 1
    assert [node.node_id for node in frontiers[0].nodes] == ["n1", "n2", "n3"]
    assert [node.target_weight for node in frontiers[0].nodes] == [1.0, 0.4, None]
    assert frontiers[0].nodes[2].target_value is None
    assert frontiers[0].nodes[2].censored is True
    assert frontiers[0].nodes[2].censor_reason == "oracle_pruned_before_followup"
    feature_map = dict(zip(PRUNER_FEATURE_NAMES, frontiers[0].nodes[0].features, strict=True))
    assert feature_map["beam_width"] == 2.0
    assert len(pairs) == 2
    assert {pair.label for pair in pairs} == {0, 1}
    assert all(pair.weight == 0.4 for pair in pairs)
    positive = next(pair for pair in pairs if pair.label == 1)
    assert positive.positive_node_id == "n2"
    assert positive.negative_node_id == "n1"
    assert positive.target_gap == 10.0


def _trace_node(
    node_id: str,
    value: float,
    *,
    root_action_id: str,
    index: int,
) -> dict:
    return {
        "node_id": node_id,
        "parent_node_id": "s:root",
        "branch_id": node_id,
        "parent_branch_id": "root",
        "frontier_index_before_prune": index,
        "kept": index < 2,
        "value": value,
        "root_action_id": root_action_id,
        "rng_id": 1,
        "decision_point_id": f"d-{node_id}",
        "depth": 1,
        "combat_depth": 1,
        "continuation_steps": 0,
        "terminal": False,
        "action_id": f"a-{node_id}",
        "action_type": "card",
        "action": {"action_id": f"a-{node_id}", "action_type": "card"},
        "policy_rank": index,
        "policy_score": None,
        "post_coverage_rank": index,
        "candidate_source": "policy",
    }


def _target(
    node_id: str,
    target_value: float,
    *,
    source: str,
    baseline: bool,
    oracle: bool,
) -> dict:
    return {
        "prune_step_id": "s:prune:0",
        "node_id": node_id,
        "root_action_id": node_id,
        "frontier_index_before_prune": 0,
        "oracle_kept": oracle,
        "target_beam_width": 2,
        "baseline_would_keep": baseline,
        "target_value": target_value,
        "target_source": source,
        "terminal_reached": source == "terminal",
        "censored": source != "terminal",
        "censor_reason": None if source == "terminal" else "value_bootstrap:max_depth",
        "best_descendant_node_id": f"leaf-{node_id}",
    }
