from __future__ import annotations

import unittest

from sts2_training.decision.candidate_coverage import CoverageConstrainedPolicy
from sts2_training.decision.policy import ActionCandidate, PolicyModel


def _action(action_id: str, action_type: str) -> dict:
    return {
        "action_id": action_id,
        "action_type": action_type,
        "is_available": True,
    }


class _RankedPolicy(PolicyModel):
    def __init__(self, action_ids: list[str]) -> None:
        self._action_ids = action_ids

    def propose(self, legal_actions, masked_emulator_dto, *, top_k):
        return [ActionCandidate(action_id=action_id) for action_id in self._action_ids[:top_k]]


class _BatchOnlyPolicy(PolicyModel):
    def propose_batch(self, requests, *, top_k):
        return [
            [ActionCandidate(action_id=action["action_id"]) for action in actions[:top_k]]
            for actions, _dto in requests
        ]


class CoverageConstrainedPolicyTest(unittest.TestCase):
    def test_retains_end_turn_and_card_even_when_ranker_omits_them(self) -> None:
        actions = [
            _action("p1", "potion"),
            _action("p2", "potion"),
            _action("p3", "potion"),
            _action("card", "card"),
            _action("end", "system"),
        ]
        policy = CoverageConstrainedPolicy(_RankedPolicy(["p1", "p2", "p3"]))

        candidates = policy.propose(actions, {}, top_k=4)
        ids = [candidate.action_id for candidate in candidates]

        self.assertIn("end", ids)
        self.assertIn("card", ids)
        self.assertIn("p1", ids)

    def test_retains_completion_branch_for_pending_card_choice(self) -> None:
        actions = [
            *[_action(f"card-{index}", "choice_card") for index in range(5)],
            _action("confirm", "choice_confirm"),
        ]
        policy = CoverageConstrainedPolicy(
            _RankedPolicy([f"card-{index}" for index in range(5)])
        )

        candidates = policy.propose(actions, {}, top_k=4)

        self.assertIn("confirm", [candidate.action_id for candidate in candidates])

    def test_batch_only_learned_policy_keeps_same_coverage_contract(self) -> None:
        actions = [
            _action("card-1", "choice_card"),
            _action("card-2", "choice_card"),
            _action("confirm", "choice_confirm"),
        ]
        policy = CoverageConstrainedPolicy(_BatchOnlyPolicy())

        candidates = policy.propose_batch([(actions, {})], top_k=2)[0]

        self.assertIn("confirm", [candidate.action_id for candidate in candidates])

    def test_top_one_does_not_force_structural_branch(self) -> None:
        actions = [_action("card", "card"), _action("end", "system")]
        policy = CoverageConstrainedPolicy(_RankedPolicy(["card"]))

        candidates = policy.propose(actions, {}, top_k=1)

        self.assertEqual([candidate.action_id for candidate in candidates], ["card"])


if __name__ == "__main__":
    unittest.main()
