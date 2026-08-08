from __future__ import annotations

import random
import unittest

from sts2_training.decision.policy import ActionCandidate, PriorHeuristicPolicy


def _action(action_id: str, action_type: str, *, is_available: bool = True) -> dict:
    return {"action_id": action_id, "action_type": action_type, "is_available": is_available}


class PriorHeuristicPolicyTest(unittest.TestCase):
    def test_prioritizes_card_over_other_categories(self) -> None:
        legal_actions = [
            _action("a-end", "system"),
            _action("a-card1", "card"),
            _action("a-card2", "card"),
            _action("a-choice", "choice_card"),
        ]
        policy = PriorHeuristicPolicy()

        candidates = policy.propose(legal_actions, {}, top_k=2)

        self.assertEqual([c.action_id for c in candidates], ["a-card1", "a-card2"])
        self.assertTrue(all(isinstance(c, ActionCandidate) for c in candidates))

    def test_falls_back_to_other_categories_when_none_of_priority_present(self) -> None:
        legal_actions = [_action("a-end", "system"), _action("a-potion", "potion")]
        policy = PriorHeuristicPolicy()

        candidates = policy.propose(legal_actions, {}, top_k=5)

        self.assertEqual({c.action_id for c in candidates}, {"a-end", "a-potion"})

    def test_excludes_unavailable_actions(self) -> None:
        legal_actions = [
            _action("a-card1", "card", is_available=False),
            _action("a-card2", "card"),
        ]
        policy = PriorHeuristicPolicy()

        candidates = policy.propose(legal_actions, {}, top_k=5)

        self.assertEqual([c.action_id for c in candidates], ["a-card2"])

    def test_empty_when_no_available_actions(self) -> None:
        legal_actions = [_action("a-card1", "card", is_available=False)]
        policy = PriorHeuristicPolicy()

        self.assertEqual(policy.propose(legal_actions, {}, top_k=5), [])

    def test_top_k_truncates(self) -> None:
        legal_actions = [_action(f"a-card{i}", "card") for i in range(5)]
        policy = PriorHeuristicPolicy()

        candidates = policy.propose(legal_actions, {}, top_k=2)

        self.assertEqual(len(candidates), 2)

    def test_rejects_non_positive_top_k(self) -> None:
        policy = PriorHeuristicPolicy()
        with self.assertRaises(ValueError):
            policy.propose([_action("a", "card")], {}, top_k=0)

    def test_rng_shuffle_is_deterministic_for_seeded_rng(self) -> None:
        legal_actions = [_action(f"a-card{i}", "card") for i in range(10)]
        policy_a = PriorHeuristicPolicy(rng=random.Random(42))
        policy_b = PriorHeuristicPolicy(rng=random.Random(42))

        result_a = [c.action_id for c in policy_a.propose(legal_actions, {}, top_k=10)]
        result_b = [c.action_id for c in policy_b.propose(legal_actions, {}, top_k=10)]

        self.assertEqual(result_a, result_b)

    def test_propose_batch_default_matches_looping_propose(self) -> None:
        legal_actions = [_action("a-card1", "card")]
        policy = PriorHeuristicPolicy()

        batched = policy.propose_batch([(legal_actions, {}), (legal_actions, {})], top_k=1)

        self.assertEqual(len(batched), 2)
        self.assertEqual(batched[0], policy.propose(legal_actions, {}, top_k=1))


if __name__ == "__main__":
    unittest.main()
