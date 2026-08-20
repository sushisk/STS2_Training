"""Turn-aligned leaf scoring (quiescence).

A fixed depth budget stops different lines at different points in the turn cycle. With
``max_depth=2``, "card, card" is still inside the player's turn with the enemy's published
attack unpaid, while "End Turn, card" already absorbed that attack and banked a fresh
turn. Comparing those directly decides the root action by turn parity, which is what kept
the search ending turns with playable cards in hand.

The scenario below is that comparison in miniature: playing a card looks better only
because its leaf has not yet paid for the enemy's attack, and paying it flips the answer.
"""

from __future__ import annotations

import unittest

from sts2_training.decision.beam_search import BeamSearchConfig, BeamSearchEngine
from sts2_training.decision.combat_decision import COMBAT_BEAM_ACTION_TYPES
from sts2_training.decision.policy import ActionCandidate, PolicyModel
from sts2_training.decision.search_trace import InMemorySearchTraceCollector
from sts2_training.decision.value import ValueModel

_ROOT_ACTIONS = [
    {"action_id": "end", "action_type": "system", "label": "End Turn", "is_available": True},
    {"action_id": "card", "action_type": "card", "label": "STRIKE", "is_available": True},
]


def _dto(*, turn: int, score: float, actions=_ROOT_ACTIONS) -> dict:
    return {
        "boundary": "stable",
        "turnNumber": turn,
        "combatRoundNumber": turn,
        "score": score,
        "legal_actions": list(actions),
    }


class _Policy(PolicyModel):
    def propose(self, legal_actions, masked_emulator_dto, *, top_k):
        del masked_emulator_dto
        return [ActionCandidate(action_id=a["action_id"]) for a in legal_actions[:top_k]]


class _Value(ValueModel):
    def evaluate(self, masked_emulator_dto):
        return float(masked_emulator_dto["score"])


class _Client:
    """Two root lines whose leaves stop in different turns.

    * ``card`` -> still turn 1, score 30 (the enemy has not attacked yet)
    * ``end``  -> turn 2, score 20 (attack absorbed, new turn in hand)

    Ending the ``card`` leaf's turn settles it at score 10, so the honest comparison is
    10 vs 20 and ``end`` should win - the opposite of the unaligned 30 vs 20.
    """

    instance_type = "combat"
    max_emulate_actions_items = 8
    pending_retry = None
    session_invalid = False

    def __init__(self) -> None:
        self.emulate_calls: list[list[dict]] = []
        self.aligned_actions: list[str] = []

    def _result_for(self, item: dict) -> dict:
        action_id = item["action_id"]
        parent = item["parent_branch_id"]
        if parent == "root":
            dto = _dto(turn=1, score=30.0) if action_id == "card" else _dto(turn=2, score=20.0)
        else:
            # Depth-2 expansion keeps each line in its own turn and score.
            parent_is_card_line = parent in self._card_line
            if action_id == "end":
                self.aligned_actions.append(parent)
                dto = _dto(turn=2, score=10.0) if parent_is_card_line else _dto(turn=3, score=5.0)
            else:
                dto = (
                    _dto(turn=1, score=30.0)
                    if parent_is_card_line
                    else _dto(turn=2, score=20.0)
                )
        return {
            "status": "completed",
            "branch_id": item["branch_id"],
            "parent_branch_id": parent,
            "rng_id": item["rng_id"],
            "decision_point_id": f"d-{item['branch_id']}",
            "branch_log": [],
            "masked_emulator_dto": dto,
        }

    _card_line: set[str] = set()

    async def emulate_actions(self, instance_id, items, *, timeout_s, simulation_options=None):
        del instance_id, timeout_s, simulation_options
        self.emulate_calls.append([dict(item) for item in items])
        results = {}
        for item in items:
            if item["parent_branch_id"] == "root" and item["action_id"] == "card":
                self._card_line.add(item["branch_id"])
            elif item["parent_branch_id"] in self._card_line:
                self._card_line.add(item["branch_id"])
            results[item["branch_id"]] = self._result_for(item)
        return {"status": "completed", "branch_results": results}

    async def cancel_branches(self, instance_id, branch_ids, *, timeout_s):
        del instance_id, timeout_s
        return {
            "status": "completed",
            "branch_statuses": {b: "cancelled" for b in branch_ids},
        }

    async def release_branches(self, instance_id, branch_ids, *, timeout_s):
        del instance_id, timeout_s
        return {
            "status": "completed",
            "branch_statuses": {b: "released" for b in branch_ids},
        }


def _engine(client, *, aligned: bool, collector=None) -> BeamSearchEngine:
    return BeamSearchEngine(
        client,
        policy=_Policy(),
        value_fn=_Value(),
        config=BeamSearchConfig(
            max_depth=2,
            beam_width=4,
            top_k_actions=2,
            beam_searchable_action_types=COMBAT_BEAM_ACTION_TYPES,
            turn_aligned_leaves=aligned,
        ),
        trace_collector=collector,
    )


_ROOT_DECISION = {
    "decision_point_id": "d-root",
    "masked_emulator_dto": _dto(turn=1, score=0.0),
}


class TurnAlignedLeavesTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _Client._card_line = set()

    async def test_unaligned_search_prefers_the_line_with_unpaid_damage(self) -> None:
        engine = _engine(_Client(), aligned=False)

        result = await engine.search("inst", _ROOT_DECISION, timeout_s=5.0)

        self.assertEqual(result.best_root_action_id, "card")
        self.assertEqual(result.best_node_score, 30.0)
        self.assertEqual(result.stats.leaves_turn_aligned, 0)

    async def test_alignment_settles_the_lagging_leaf_and_flips_the_choice(self) -> None:
        engine = _engine(_Client(), aligned=True)

        result = await engine.search("inst", _ROOT_DECISION, timeout_s=5.0)

        self.assertEqual(result.best_root_action_id, "end")
        self.assertEqual(result.best_node_score, 20.0)
        self.assertGreater(result.stats.leaves_turn_aligned, 0)

    async def test_alignment_is_recorded_on_the_end_trace(self) -> None:
        collector = InMemorySearchTraceCollector()
        engine = _engine(_Client(), aligned=True, collector=collector)

        result = await engine.search("inst", _ROOT_DECISION, timeout_s=5.0)

        ends = [e for e in collector.events if getattr(e, "event_type", "") == "search_end"]
        self.assertEqual(
            [end.leaves_turn_aligned for end in ends], [result.stats.leaves_turn_aligned]
        )

    async def test_alignment_costs_one_extra_batch(self) -> None:
        plain = _Client()
        await _engine(plain, aligned=False).search("inst", _ROOT_DECISION, timeout_s=5.0)
        _Client._card_line = set()
        aligned = _Client()
        await _engine(aligned, aligned=True).search("inst", _ROOT_DECISION, timeout_s=5.0)

        self.assertEqual(len(aligned.emulate_calls), len(plain.emulate_calls) + 1)


class _AlreadyAlignedClient(_Client):
    """Every leaf stops in the same turn, so there is nothing to settle."""

    def _result_for(self, item: dict) -> dict:
        result = super()._result_for(item)
        result["masked_emulator_dto"]["turnNumber"] = 1
        result["masked_emulator_dto"]["combatRoundNumber"] = 1
        return result


class _FaultingAlignmentClient(_Client):
    """The alignment batch faults; the unaligned scores must still stand."""

    async def emulate_actions(self, instance_id, items, *, timeout_s, simulation_options=None):
        if len(self.emulate_calls) >= 2:  # the alignment batch follows the two depth batches
            self.emulate_calls.append([dict(item) for item in items])
            return {"status": "completed", "branch_results": {}}
        return await super().emulate_actions(
            instance_id, items, timeout_s=timeout_s, simulation_options=simulation_options
        )


class AlignmentEdgeCaseTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _Client._card_line = set()

    async def test_no_extra_request_when_every_leaf_is_already_in_the_same_turn(self) -> None:
        client = _AlreadyAlignedClient()
        engine = _engine(client, aligned=True)

        result = await engine.search("inst", _ROOT_DECISION, timeout_s=5.0)

        self.assertEqual(result.stats.leaves_turn_aligned, 0)
        self.assertEqual(len(client.emulate_calls), 2)

    async def test_a_faulting_alignment_batch_leaves_the_original_scores(self) -> None:
        client = _FaultingAlignmentClient()
        engine = _engine(client, aligned=True)

        result = await engine.search("inst", _ROOT_DECISION, timeout_s=5.0)

        self.assertEqual(result.stats.leaves_turn_aligned, 0)
        self.assertEqual(result.best_root_action_id, "card")
        self.assertEqual(result.best_node_score, 30.0)


if __name__ == "__main__":
    unittest.main()
