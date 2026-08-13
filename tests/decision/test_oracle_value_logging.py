from __future__ import annotations

import unittest

from sts2_training.decision.oracle_search import (
    OracleRngOutcome,
    OracleTargetMetadata,
    OracleTargets,
    RootActionOracleTarget,
)
from sts2_training.decision.oracle_value_logging import build_root_action_value_samples
from sts2_training.decision.search_trace import ResolvedNodeTrace


def _resolved(
    *,
    node_id: str,
    parent_node_id: str,
    branch_id: str,
    parent_branch_id: str,
    combat_depth: int,
    depth: int,
    value: float,
    root_action_id: str = "a",
    rng_id: int = 7,
    decision_point_id: str | None = None,
    terminal: bool = False,
) -> ResolvedNodeTrace:
    return ResolvedNodeTrace(
        search_id="s",
        node_id=node_id,
        parent_node_id=parent_node_id,
        branch_id=branch_id,
        parent_branch_id=parent_branch_id,
        root_action_id=root_action_id,
        rng_id=rng_id,
        decision_point_id=(
            f"decision-{node_id}" if decision_point_id is None else decision_point_id
        ),
        depth=depth,
        combat_depth=combat_depth,
        continuation_steps=0,
        value=value,
        value_is_fresh=True,
        value_source="terminal" if terminal else "value_bootstrap",
        state_kind="terminal" if terminal else "stable",
        resolution="terminal" if terminal else "expandable_stable",
        terminal=terminal,
        action_id="a",
        action_type="card",
        action={"action_id": "a", "action_type": "card"},
        policy_rank=0,
        policy_score=1.0,
        post_coverage_rank=0,
        candidate_source="policy",
    )


def _metadata(*, search_reason: str = "max_depth") -> OracleTargetMetadata:
    return OracleTargetMetadata(
        search_id="s",
        oracle_beam_width=16,
        target_beam_width=4,
        top_k_actions=8,
        max_depth=4,
        max_continuation_steps=8,
        time_budget_ms=None,
        exhaustive_root_actions=True,
        rng_sampling="independent",
        search_reason=search_reason,
        pruner_name="value_top_k",
        pruner_version="1",
    )


class RootActionValueLoggingTest(unittest.TestCase):
    def test_joins_only_root_state_dto_to_deeper_oracle_target(self) -> None:
        root_state = _resolved(
            node_id="s:root-state",
            parent_node_id="s:root",
            branch_id="root-state",
            parent_branch_id="root",
            combat_depth=1,
            depth=1,
            value=1.5,
        )
        deep_leaf = _resolved(
            node_id="s:deep-leaf",
            parent_node_id=root_state.node_id,
            branch_id="deep-leaf",
            parent_branch_id=root_state.branch_id,
            combat_depth=2,
            depth=2,
            value=9.0,
        )
        targets = OracleTargets(
            metadata=_metadata(),
            root_actions=(
                RootActionOracleTarget(
                    action_id="a",
                    action={"action_id": "a", "action_type": "card"},
                    evaluated=True,
                    estimated_q=9.0,
                    rng_outcomes=(
                        OracleRngOutcome(
                            rng_id=7,
                            value=9.0,
                            target_source="value_bootstrap",
                            terminal_reached=False,
                            deepest_combat_depth=2,
                            censored=True,
                            censor_reason="value_bootstrap:max_depth",
                            best_node_id=deep_leaf.node_id,
                        ),
                    ),
                    target_source="value_bootstrap",
                    terminal_reached=False,
                    censored=True,
                    censor_reason="value_bootstrap:max_depth",
                ),
            ),
            stable_nodes=(),
        )
        raw_root_dto = {
            "hp": 42,
            "energy": 2,
            "legal_actions": [{"action_id": "next", "action_type": "card"}],
        }

        samples = build_root_action_value_samples(
            [root_state, deep_leaf],
            targets,
            root_state_dtos={root_state.node_id: raw_root_dto},
        )

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample.action_id, "a")
        self.assertEqual(sample.root_state_node_id, root_state.node_id)
        self.assertEqual(sample.decision_point_id, f"decision-{root_state.node_id}")
        self.assertEqual(sample.masked_emulator_dto, raw_root_dto)
        self.assertEqual(sample.target_value, 9.0)
        self.assertEqual(sample.best_node_id, deep_leaf.node_id)
        self.assertEqual(sample.target_source, "value_bootstrap")

    def test_terminal_root_state_normalizes_missing_next_decision_id_to_none(self) -> None:
        root_state = _resolved(
            node_id="s:terminal-root-state",
            parent_node_id="s:root",
            branch_id="terminal-root-state",
            parent_branch_id="root",
            combat_depth=1,
            depth=1,
            value=100.0,
            decision_point_id="",
            terminal=True,
        )
        targets = OracleTargets(
            metadata=_metadata(search_reason="terminal"),
            root_actions=(
                RootActionOracleTarget(
                    action_id="a",
                    action={"action_id": "a", "action_type": "card"},
                    evaluated=True,
                    estimated_q=100.0,
                    rng_outcomes=(
                        OracleRngOutcome(
                            rng_id=7,
                            value=100.0,
                            target_source="terminal",
                            terminal_reached=True,
                            deepest_combat_depth=1,
                            censored=False,
                            censor_reason=None,
                            best_node_id=root_state.node_id,
                        ),
                    ),
                    target_source="terminal",
                    terminal_reached=True,
                    censored=False,
                    censor_reason=None,
                ),
            ),
            stable_nodes=(),
        )
        raw_root_dto = {
            "terminal": True,
            "outcome": "victory",
            "legal_actions": [],
        }

        samples = build_root_action_value_samples(
            [root_state],
            targets,
            root_state_dtos={root_state.node_id: raw_root_dto},
        )

        self.assertEqual(len(samples), 1)
        self.assertIsNone(samples[0].decision_point_id)
        self.assertEqual(samples[0].target_source, "terminal")
        self.assertEqual(samples[0].target_value, 100.0)
        self.assertTrue(samples[0].terminal_reached)
        self.assertEqual(samples[0].masked_emulator_dto, raw_root_dto)

    def test_keeps_censored_root_state_with_null_target_for_audit(self) -> None:
        root_state = _resolved(
            node_id="s:root-state",
            parent_node_id="s:root",
            branch_id="root-state",
            parent_branch_id="root",
            combat_depth=1,
            depth=1,
            value=1.5,
        )
        targets = OracleTargets(
            metadata=_metadata(search_reason="time_budget"),
            root_actions=(
                RootActionOracleTarget(
                    action_id="a",
                    action={"action_id": "a", "action_type": "card"},
                    evaluated=True,
                    estimated_q=None,
                    rng_outcomes=(
                        OracleRngOutcome(
                            rng_id=7,
                            value=None,
                            target_source="no_target",
                            terminal_reached=False,
                            deepest_combat_depth=1,
                            censored=True,
                            censor_reason="search_ended_before_followup:time_budget",
                            best_node_id=None,
                        ),
                    ),
                    target_source="no_target",
                    terminal_reached=False,
                    censored=True,
                    censor_reason="search_ended_before_followup:time_budget",
                ),
            ),
            stable_nodes=(),
        )

        samples = build_root_action_value_samples(
            [root_state],
            targets,
            root_state_dtos={root_state.node_id: {"hp": 42}},
        )

        self.assertEqual(len(samples), 1)
        self.assertIsNone(samples[0].target_value)
        self.assertTrue(samples[0].censored)
        self.assertEqual(samples[0].target_source, "no_target")


if __name__ == "__main__":
    unittest.main()
