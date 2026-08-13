from __future__ import annotations

import unittest

from sts2_training.decision.beam_search import (
    BeamSearchEngine,
    BeamSearchResult,
    BeamSearchStats,
)
from sts2_training.decision.oracle_log import (
    ORACLE_VALUE_MASK_VERSION,
    oracle_collection_record,
)
from sts2_training.decision.oracle_search import (
    BudgetedOracleCollector,
    OracleCollectionResult,
    OracleProvenance,
    _OracleTraceCollector,
    _effective_time_budget_ms,
    build_oracle_targets,
)
from sts2_training.decision.policy import PolicyModel
from sts2_training.decision.search_trace import (
    PolicyProposalTrace,
    SearchTraceEnd,
    SearchTraceStart,
)
from sts2_training.decision.value import ValueModel


class _NoopPolicy(PolicyModel):
    def propose(self, legal_actions, masked_emulator_dto, *, top_k):
        return []


class _NoopValue(ValueModel):
    def evaluate_batch(self, masked_emulator_dtos):
        return [0.0 for _dto in masked_emulator_dtos]


class OracleRngNamespaceTest(unittest.TestCase):
    def test_from_beam_engine_shares_rng_namespace_across_oracle_runtime_oracle(self) -> None:
        runtime = BeamSearchEngine(
            object(),
            policy=_NoopPolicy(),
            value_fn=_NoopValue(),
        )
        oracle = BudgetedOracleCollector.from_beam_engine(runtime)

        oracle_first = oracle._branch_allocator.next_rng_id()  # noqa: SLF001
        runtime_next = runtime._allocator.next_rng_id()  # noqa: SLF001
        oracle_second = oracle._branch_allocator.next_rng_id()  # noqa: SLF001

        self.assertEqual((oracle_first, runtime_next, oracle_second), (1, 2, 3))
        self.assertEqual(len({oracle_first, runtime_next, oracle_second}), 3)
        self.assertIs(oracle._branch_allocator, runtime._allocator)  # noqa: SLF001

    def test_explicit_oracle_collector_keeps_allocator_across_collections(self) -> None:
        oracle = BudgetedOracleCollector(
            object(),
            policy=_NoopPolicy(),
            value_fn=_NoopValue(),
        )

        first = oracle._branch_allocator.next_rng_id()  # noqa: SLF001
        second = oracle._branch_allocator.next_rng_id()  # noqa: SLF001

        self.assertEqual((first, second), (1, 2))


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
        dto = {
            "mask_version": ORACLE_VALUE_MASK_VERSION,
            "dto_version": "emulator-test",
            "legal_actions": [],
        }
        record = oracle_collection_record(
            {"decision_point_id": "d-root", "masked_emulator_dto": dto},
            result,
            instance_id="inst",
            decision_index=0,
            runtime_transition={
                "chosen_action_id": "a",
                "chosen_action": {"action_id": "a"},
                "next_decision_point_id": "d-next",
                "commit_response_metadata": {"decision_point_id": "d-next"},
                "next_masked_emulator_dto": dto,
            },
        )

        self.assertEqual(record["search_trace"][0]["time_budget_ms"], 5000.0)
        self.assertEqual(
            record["oracle_targets"]["metadata"]["time_budget_ms"],
            5000.0,
        )


if __name__ == "__main__":
    unittest.main()
