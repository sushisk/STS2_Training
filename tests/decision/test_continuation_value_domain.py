from __future__ import annotations

import unittest

from sts2_training.decision.beam_search import BeamNode, BeamSearchConfig, BeamSearchEngine
from sts2_training.decision.combat_decision import COMBAT_BEAM_ACTION_TYPES
from sts2_training.decision.policy import ActionCandidate, PolicyModel
from sts2_training.decision.value import ValueModel


class _UnusedPolicy(PolicyModel):
    pass


class _RecordingValue(ValueModel):
    def __init__(self) -> None:
        self.batches: list[list[dict]] = []

    def evaluate_batch(self, dtos, *, _unused=None):
        copied = [dict(dto) for dto in dtos]
        self.batches.append(copied)
        return [42.0 for _ in dtos]


class ContinuationValueDomainTest(unittest.TestCase):
    def _engine(self, value: _RecordingValue) -> BeamSearchEngine:
        return BeamSearchEngine(
            object(),
            policy=_UnusedPolicy(),
            value_fn=value,
            config=BeamSearchConfig(
                beam_searchable_action_types=COMBAT_BEAM_ACTION_TYPES,
                max_depth=2,
            ),
        )

    def test_pending_continuation_state_is_not_sent_to_value_model(self) -> None:
        value = _RecordingValue()
        engine = self._engine(value)
        parent = BeamNode(
            branch_id="root",
            parent_branch_id="root",
            rng_id=0,
            decision_point_id="d0",
            masked_emulator_dto={
                "legal_actions": [
                    {"action_id": "pick", "action_type": "choice_card", "is_available": True}
                ]
            },
            depth=0,
            value=7.0,
            root_action_id=None,
        )
        branch_results = {
            "b1": {
                "status": "completed",
                "decision_point_id": "d1",
                "masked_emulator_dto": {
                    "pendingChoice": {"selectedCount": 1},
                    "legal_actions": [
                        {"action_id": "pick-2", "action_type": "choice_card", "is_available": True},
                        {"action_id": "confirm", "action_type": "choice_confirm", "is_available": True},
                    ],
                },
            }
        }

        next_beam, finished, _ms, _depth, _limit = engine._score_frontier(  # noqa: SLF001
            [(parent, ActionCandidate("pick"), "b1", 1)], branch_results
        )

        self.assertEqual(value.batches, [])
        self.assertEqual(finished, [])
        self.assertEqual(len(next_beam), 1)
        self.assertEqual(next_beam[0].value, 7.0)

    def test_resolved_stable_state_is_sent_to_value_model(self) -> None:
        value = _RecordingValue()
        engine = self._engine(value)
        parent = BeamNode(
            branch_id="b0",
            parent_branch_id="root",
            rng_id=1,
            decision_point_id="d0",
            masked_emulator_dto={
                "legal_actions": [
                    {"action_id": "confirm", "action_type": "choice_confirm", "is_available": True}
                ]
            },
            depth=1,
            value=7.0,
            root_action_id="play-card",
            combat_depth=1,
            continuation_steps=1,
        )
        stable_dto = {
            "hp": 40,
            "legal_actions": [
                {"action_id": "strike", "action_type": "card", "is_available": True},
                {"action_id": "end", "action_type": "system", "is_available": True},
            ],
        }
        branch_results = {
            "b1": {
                "status": "completed",
                "decision_point_id": "d1",
                "masked_emulator_dto": stable_dto,
            }
        }

        _next, finished, _ms, hit_depth, _limit = engine._score_frontier(  # noqa: SLF001
            [(parent, ActionCandidate("confirm"), "b1", 1)], branch_results
        )

        self.assertEqual(value.batches, [[stable_dto]])
        self.assertTrue(hit_depth)
        self.assertEqual(finished[0].value, 42.0)


if __name__ == "__main__":
    unittest.main()
