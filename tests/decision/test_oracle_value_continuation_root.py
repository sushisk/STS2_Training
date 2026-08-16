from __future__ import annotations

import unittest

from sts2_training.api.contract import ROOT_BRANCH_ID
from sts2_training.decision.beam_search import BeamNode, BeamSearchConfig
from sts2_training.decision.candidate_coverage import CandidateProposal
from sts2_training.decision.oracle_search import (
    OracleRngOutcome,
    OracleTargetMetadata,
    OracleTargets,
    RootActionOracleTarget,
    _OracleTraceCollector,
)
from sts2_training.decision.oracle_value_logging import (
    _RootValueCapturingOracleEngine,
    build_root_action_value_samples,
)
from sts2_training.decision.policy import ActionCandidate, PolicyModel
from sts2_training.decision.search_trace import SearchTraceStart
from sts2_training.decision.value import ValueModel


class _NoopPolicy(PolicyModel):
    def propose(self, legal_actions, masked_emulator_dto, *, top_k):
        return []


class _ConstantValue(ValueModel):
    def evaluate_batch(self, dtos):
        return [float(index + 1) for index, _dto in enumerate(dtos)]


def _proposal(action_id: str) -> CandidateProposal:
    return CandidateProposal(
        candidate=ActionCandidate(action_id),
        policy_rank=0,
        policy_score=1.0,
        post_coverage_rank=0,
        candidate_source="policy",
    )


class ContinuationRootValueSamplingTest(unittest.TestCase):
    def test_first_fresh_continuation_root_state_is_retained_at_combat_depth_zero(self) -> None:
        trace = _OracleTraceCollector(
            descendant_top_k=2,
            exhaustive_root_actions=True,
            effective_time_budget_ms=1000.0,
        )
        trace.record(
            SearchTraceStart(
                search_id="s",
                instance_id="inst",
                root_decision_point_id="d-root",
                beam_width=2,
                top_k_actions=2,
                max_depth=2,
                max_continuation_steps=4,
                time_budget_ms=1000.0,
                pruner_name="value_top_k",
                pruner_version="1",
            )
        )
        engine = _RootValueCapturingOracleEngine(
            object(),
            policy=_NoopPolicy(),
            value_fn=_ConstantValue(),
            config=BeamSearchConfig(beam_width=2, top_k_actions=2, max_depth=2),
            trace_collector=trace,
            exhaustive_root_actions=True,
        )
        root = BeamNode(
            branch_id=ROOT_BRANCH_ID,
            parent_branch_id=ROOT_BRANCH_ID,
            rng_id=0,
            decision_point_id="d-root",
            masked_emulator_dto={
                "legal_actions": [
                    {"action_id": "pick", "action_type": "choice_card", "is_available": True}
                ]
            },
            depth=0,
            value=0.0,
            root_action_id=None,
        )
        first_result = {
            "b1": {
                "status": "completed",
                "decision_point_id": "d-stable",
                "masked_emulator_dto": {
                    "hp": 40,
                    "legal_actions": [
                        {"action_id": "attack", "action_type": "card", "is_available": True}
                    ],
                },
            }
        }

        next_beam, finished, _value_ms, _hit_depth, _hit_limit, _faulted = engine._score_frontier(  # noqa: SLF001
            [(root, _proposal("pick"), "b1", 7)],
            first_result,
            search_id="s",
        )

        self.assertEqual(len(next_beam), 1)
        self.assertEqual(finished, [])
        root_state = next_beam[0]
        self.assertEqual(root_state.root_action_id, "pick")
        self.assertEqual(root_state.combat_depth, 0)
        self.assertEqual(set(engine.root_state_dtos), {"s:b1"})

        deep_result = {
            "b2": {
                "status": "completed",
                "decision_point_id": "d-deep",
                "masked_emulator_dto": {
                    "hp": 39,
                    "legal_actions": [
                        {"action_id": "end", "action_type": "system", "is_available": True}
                    ],
                },
            }
        }
        deeper_beam, _finished, _value_ms, _hit_depth, _hit_limit, _faulted = engine._score_frontier(  # noqa: SLF001
            [(root_state, _proposal("attack"), "b2", 7)],
            deep_result,
            search_id="s",
        )

        self.assertEqual(len(deeper_beam), 1)
        self.assertEqual(deeper_beam[0].combat_depth, 1)
        self.assertEqual(set(engine.root_state_dtos), {"s:b1"})

        targets = OracleTargets(
            metadata=OracleTargetMetadata(
                search_id="s",
                oracle_beam_width=2,
                target_beam_width=1,
                top_k_actions=2,
                max_depth=2,
                max_continuation_steps=4,
                time_budget_ms=1000.0,
                exhaustive_root_actions=True,
                rng_sampling="independent",
                search_reason="max_depth",
                pruner_name="value_top_k",
                pruner_version="1",
            ),
            root_actions=(
                RootActionOracleTarget(
                    action_id="pick",
                    action={"action_id": "pick", "action_type": "choice_card"},
                    evaluated=True,
                    estimated_q=1.0,
                    rng_outcomes=(
                        OracleRngOutcome(
                            rng_id=7,
                            value=1.0,
                            target_source="value_bootstrap",
                            terminal_reached=False,
                            deepest_combat_depth=1,
                            censored=True,
                            censor_reason="value_bootstrap:max_depth",
                            best_node_id="s:b2",
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
            trace.events,
            targets,
            root_state_dtos=engine.root_state_dtos,
        )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].action_id, "pick")
        self.assertEqual(samples[0].root_state_node_id, "s:b1")
        self.assertEqual(samples[0].decision_point_id, "d-stable")
        self.assertEqual(samples[0].masked_emulator_dto["hp"], 40)
        self.assertEqual(samples[0].target_value, 1.0)
        self.assertEqual(samples[0].best_node_id, "s:b2")


if __name__ == "__main__":
    unittest.main()
