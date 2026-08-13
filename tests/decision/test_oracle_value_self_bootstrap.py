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


class RootValueSelfBootstrapTest(unittest.TestCase):
    def test_nonterminal_root_self_bootstrap_becomes_no_target(self) -> None:
        node = ResolvedNodeTrace(
            search_id="s",
            node_id="s:root-state",
            parent_node_id="s:root",
            branch_id="root-state",
            parent_branch_id="root",
            root_action_id="a",
            rng_id=7,
            decision_point_id="after-a",
            depth=1,
            combat_depth=1,
            continuation_steps=0,
            value=1.5,
            value_is_fresh=True,
            value_source="value_bootstrap",
            state_kind="stable",
            resolution="max_depth",
            terminal=False,
            action_id="a",
            action_type="card",
            action={"action_id": "a", "action_type": "card"},
            policy_rank=0,
            policy_score=1.0,
            post_coverage_rank=0,
            candidate_source="policy",
        )
        metadata = OracleTargetMetadata(
            search_id="s",
            oracle_beam_width=8,
            target_beam_width=4,
            top_k_actions=8,
            max_depth=1,
            max_continuation_steps=8,
            time_budget_ms=None,
            exhaustive_root_actions=True,
            rng_sampling="independent",
            search_reason="max_depth",
            pruner_name="value_top_k",
            pruner_version="1",
        )
        targets = OracleTargets(
            metadata=metadata,
            root_actions=(
                RootActionOracleTarget(
                    action_id="a",
                    action={"action_id": "a", "action_type": "card"},
                    evaluated=True,
                    estimated_q=1.5,
                    rng_outcomes=(
                        OracleRngOutcome(
                            rng_id=7,
                            value=1.5,
                            target_source="value_bootstrap",
                            terminal_reached=False,
                            deepest_combat_depth=1,
                            censored=True,
                            censor_reason="value_bootstrap:max_depth",
                            best_node_id=node.node_id,
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

        samples = build_root_action_value_samples(
            [node],
            targets,
            root_state_dtos={node.node_id: {"hp": 42}},
        )

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertIsNone(sample.target_value)
        self.assertEqual(sample.target_source, "no_target")
        self.assertTrue(sample.censored)
        self.assertEqual(
            sample.censor_reason,
            "root_state_self_bootstrap_not_deeper_oracle_target",
        )
        self.assertIsNone(sample.best_node_id)


if __name__ == "__main__":
    unittest.main()
