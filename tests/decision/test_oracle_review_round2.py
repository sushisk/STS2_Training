from __future__ import annotations

import unittest

from sts2_training.decision.beam_search import BeamSearchResult, BeamSearchStats
from sts2_training.decision.oracle_log import oracle_collection_record
from sts2_training.decision.oracle_search import (
    OracleCollectionResult,
    OracleProvenance,
    _OracleTraceCollector,
    _effective_time_budget_ms,
    build_oracle_targets,
)
from sts2_training.decision.search_trace import (
    PolicyProposalTrace,
    SearchTraceEnd,
    SearchTraceStart,
)


class OracleEffectiveBudgetMetadataTest(unittest.TestCase):
    def _start(self, time_budget_ms: float | None) -> SearchTraceStart:
        return SearchTraceStart(
            search_id="search",
            instance_id="inst",
            root_decision_point_id="d-root",
            beam_width=1,
            top_k_actions=1,
            max_depth=1,
            max_continuation_steps=1,
            time_budget_ms=time_budget_ms,
            pruner_name="value_top_k",
            pruner_version="1",
        )

    def _proposal(self) -> PolicyProposalTrace:
        return PolicyProposalTrace(
            search_id="search",
            proposal_step_id="proposal",
            parent_node_id="search:root",
            parent_branch_id="root",
            decision_point_id="d-root",
            legal_actions=(),
            candidates=(),
        )

    def test_outer_timeout_becomes_effective_budget_when_config_has_none(self) -> None:
        self.assertEqual(_effective_time_budget_ms(5.0, None), 5000.0)

    def test_config_budget_wins_when_it_is_narrower_than_outer_timeout(self) -> None:
        self.assertEqual(_effective_time_budget_ms(120.0, 2500.0), 2500.0)

    def test_effective_budget_is_persisted_in_trace_targets_and_json_record(self) -> None:
        collector = _OracleTraceCollector(
            descendant_top_k=1,
            exhaustive_root_actions=True,
            effective_time_budget_ms=5000.0,
        )
        collector.record(self._start(None))
        collector.record(self._proposal())
        collector.record(
            SearchTraceEnd(
                search_id="search",
                reason="no_candidates",
                best_root_action_id=None,
                best_value=None,
                depths_completed=0,
                nodes_expanded=0,
                branches_created=0,
            )
        )

        start = collector.events[0]
        self.assertIsInstance(start, SearchTraceStart)
        self.assertEqual(start.time_budget_ms, 5000.0)

        targets = build_oracle_targets(
            collector.events,
            target_beam_width=1,
            exhaustive_root_actions=True,
        )
        self.assertEqual(targets.metadata.time_budget_ms, 5000.0)

        result = OracleCollectionResult(
            search_result=BeamSearchResult(
                best_root_action_id=None,
                best_value=None,
                best_node=None,
                reason="no_candidates",
                stats=BeamSearchStats(),
            ),
            trace=tuple(collector.events),
            targets=targets,
            provenance=OracleProvenance(
                teacher_policy_class="test.Policy",
                teacher_inner_policy_class="test.Policy",
                teacher_coverage_policy_class=None,
                teacher_value_class="test.Value",
            ),
        )
        record = oracle_collection_record(
            {
                "decision_point_id": "d-root",
                "masked_emulator_dto": {"legal_actions": []},
            },
            result,
        )

        self.assertEqual(record["search_trace"][0]["time_budget_ms"], 5000.0)
        self.assertEqual(
            record["oracle_targets"]["metadata"]["time_budget_ms"],
            5000.0,
        )


if __name__ == "__main__":
    unittest.main()
