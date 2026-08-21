"""Turn-boundary leaf scoring: keep the search inside one turn, settle every leaf.

`_align_leaf_turns` lifted only the lagging leaves because lines that had already played
End Turn sat a half-move further on and could not be uncrossed. Keeping End Turn out of
the expandable set removes those lines entirely, so every leaf is "k cards played this
turn" and all of them can be settled at the same boundary.

The scenario below is the comparison in miniature: playing more cards looks better while
the enemy's attack is unpaid, and paying it reverses the order.
"""

from __future__ import annotations

import unittest

from sts2_training.api.contract import RequestRejectedError
from sts2_training.decision.beam_search import BeamSearchConfig, BeamSearchEngine
from sts2_training.decision.combat_decision import COMBAT_BEAM_ACTION_TYPES
from sts2_training.decision.policy import ActionCandidate, PolicyModel
from sts2_training.decision.value import ValueModel

_END = {"action_id": "end", "action_type": "system", "label": "End Turn", "is_available": True}
_CARD = {"action_id": "card", "action_type": "card", "label": "STRIKE", "is_available": True}

# Mid-turn the deeper line looks better; once the enemy has moved it is the worst.
_UNSETTLED = {0: 0.0, 1: 30.0, 2: 40.0}
_SETTLED_END_WINS = {0: 20.0, 1: 8.0, 2: 5.0}
_SETTLED_CARDS_WIN = {0: 5.0, 1: 8.0, 2: 20.0}


def _dto(*, turn: int, score: float, cards: int) -> dict:
    actions = [_END] if cards >= 2 else [_END, _CARD]
    return {
        "boundary": "stable",
        "turnNumber": turn,
        "combatRoundNumber": turn,
        "score": score,
        "cards": cards,
        "legal_actions": [dict(a) for a in actions],
    }


class _Policy(PolicyModel):
    def propose(self, legal_actions, masked_emulator_dto, *, top_k):
        del masked_emulator_dto
        return [ActionCandidate(action_id=a["action_id"]) for a in legal_actions[:top_k]]


class _Value(ValueModel):
    def evaluate(self, masked_emulator_dto):
        return float(masked_emulator_dto["score"])


class _Client:
    """One turn of play. Cards stay in turn 1; End Turn settles the line in turn 2."""

    instance_type = "combat"
    max_emulate_actions_items = 16
    pending_retry = None
    session_invalid = False
    settled = _SETTLED_END_WINS

    def __init__(self) -> None:
        self.emulate_calls: list[list[dict]] = []
        self.expanded_actions: list[str] = []
        self.cards_by_branch: dict[str, int] = {"root": 0}

    def _result_for(self, item: dict) -> dict:
        played = self.cards_by_branch[item["parent_branch_id"]]
        if item["action_id"] == "end":
            dto = _dto(turn=2, score=self.settled[played], cards=played)
        else:
            played += 1
            dto = _dto(turn=1, score=_UNSETTLED[played], cards=played)
        self.cards_by_branch[item["branch_id"]] = played
        return {
            "status": "completed",
            "branch_id": item["branch_id"],
            "parent_branch_id": item["parent_branch_id"],
            "rng_id": item["rng_id"],
            "decision_point_id": f"d-{item['branch_id']}",
            "branch_log": [],
            "masked_emulator_dto": dto,
        }

    async def emulate_actions(self, instance_id, items, *, timeout_s, simulation_options=None):
        del instance_id, timeout_s, simulation_options
        self.emulate_calls.append([dict(item) for item in items])
        results = {}
        for item in items:
            self.expanded_actions.append(item["action_id"])
            results[item["branch_id"]] = self._result_for(item)
        return {"status": "completed", "branch_results": results}

    async def cancel_branches(self, instance_id, branch_ids, *, timeout_s):
        del instance_id, timeout_s
        return {"status": "completed", "branch_statuses": {b: "cancelled" for b in branch_ids}}

    async def release_branches(self, instance_id, branch_ids, *, timeout_s):
        del instance_id, timeout_s
        return {"status": "completed", "branch_statuses": {b: "released" for b in branch_ids}}


_ROOT_DECISION = {
    "decision_point_id": "d-root",
    "masked_emulator_dto": _dto(turn=1, score=0.0, cards=0),
}


def _engine(client, *, boundary: bool) -> BeamSearchEngine:
    return BeamSearchEngine(
        client,
        policy=_Policy(),
        value_fn=_Value(),
        config=BeamSearchConfig(
            max_depth=2,
            beam_width=4,
            top_k_actions=2,
            beam_searchable_action_types=COMBAT_BEAM_ACTION_TYPES,
            turn_boundary_scoring=boundary,
        ),
    )


class TurnBoundaryScoringTest(unittest.IsolatedAsyncioTestCase):
    async def test_without_it_the_unpaid_line_wins(self) -> None:
        result = await _engine(_Client(), boundary=False).search(
            "inst", _ROOT_DECISION, timeout_s=5.0
        )

        self.assertEqual(result.best_root_action_id, "card")
        self.assertEqual(result.best_node_score, 40.0)

    async def test_settling_every_leaf_flips_the_choice_to_end_turn(self) -> None:
        result = await _engine(_Client(), boundary=True).search(
            "inst", _ROOT_DECISION, timeout_s=5.0
        )

        self.assertEqual(result.best_root_action_id, "end")
        self.assertEqual(result.best_node_score, 20.0)

    async def test_cards_still_win_when_the_settled_order_says_so(self) -> None:
        client = _Client()
        client.settled = _SETTLED_CARDS_WIN

        result = await _engine(client, boundary=True).search(
            "inst", _ROOT_DECISION, timeout_s=5.0
        )

        self.assertEqual(result.best_root_action_id, "card")
        self.assertEqual(result.best_node_score, 20.0)

    async def test_end_turn_is_never_expanded_during_the_search(self) -> None:
        client = _Client()

        await _engine(client, boundary=True).search("inst", _ROOT_DECISION, timeout_s=5.0)

        # Settling runs per ply, so batches interleave. Every batch is therefore either a
        # settling batch (all End Turn) or a depth batch (no End Turn) - never a mix,
        # which is what "End Turn is not an expandable action" means in practice.
        self.assertTrue(client.emulate_calls)
        settling = 0
        for batch in client.emulate_calls:
            action_ids = [item["action_id"] for item in batch]
            if all(action_id == "end" for action_id in action_ids):
                settling += 1
            else:
                self.assertNotIn("end", action_ids)
        self.assertGreater(settling, 0)

    async def test_the_winning_leaf_sits_past_the_turn_boundary(self) -> None:
        client = _Client()

        result = await _engine(client, boundary=True).search(
            "inst", _ROOT_DECISION, timeout_s=5.0
        )

        self.assertIsNotNone(result.best_node)
        self.assertEqual(result.best_node.masked_emulator_dto["turnNumber"], 2)
        self.assertGreater(result.stats.leaves_turn_aligned, 0)


class ExhaustedNodeTest(unittest.IsolatedAsyncioTestCase):
    async def test_a_node_with_only_end_turn_becomes_a_leaf(self) -> None:
        """Two cards exhaust the hand; the resulting node offers only End Turn.

        Before per-node leaf handling that emptied the whole proposal batch and aborted
        the search with `no_candidates`, losing every line.
        """
        client = _Client()
        client.settled = _SETTLED_CARDS_WIN

        result = await _engine(client, boundary=True).search(
            "inst", _ROOT_DECISION, timeout_s=5.0
        )

        self.assertIsNotNone(result.best_root_action_id)
        self.assertEqual(result.best_node_score, 20.0)


class _FaultingSettleClient(_Client):
    """The settling batch is rejected; the unsettled scores must still stand."""

    async def emulate_actions(self, instance_id, items, *, timeout_s, simulation_options=None):
        if all(item["action_id"] == "end" for item in items):
            self.emulate_calls.append([dict(item) for item in items])
            return {"status": "completed", "branch_results": {}}
        return await super().emulate_actions(
            instance_id, items, timeout_s=timeout_s, simulation_options=simulation_options
        )


class BestEffortTest(unittest.IsolatedAsyncioTestCase):
    async def test_a_rejected_settle_batch_leaves_the_original_scores(self) -> None:
        result = await _engine(_FaultingSettleClient(), boundary=True).search(
            "inst", _ROOT_DECISION, timeout_s=5.0
        )

        self.assertEqual(result.best_root_action_id, "card")
        self.assertEqual(result.best_node_score, 40.0)
        self.assertEqual(result.stats.leaves_turn_aligned, 0)


class _SmallBatchClient(_Client):
    """Server accepts one item per call, so the settling batch must be chunked."""

    max_emulate_actions_items = 1


class BatchLimitTest(unittest.IsolatedAsyncioTestCase):
    async def test_the_settling_batch_respects_the_server_batch_limit(self) -> None:
        client = _SmallBatchClient()

        result = await _engine(client, boundary=True).search(
            "inst", _ROOT_DECISION, timeout_s=5.0
        )

        self.assertTrue(all(len(batch) <= 1 for batch in client.emulate_calls))
        self.assertEqual(result.best_root_action_id, "end")
        self.assertEqual(result.best_node_score, 20.0)


class _CappedWholeRunClient(_Client):
    """A Whole Run server that refuses to hold more than `capacity` active branches.

    This is the shape that made the mode a silent no-op in the first Whole Run
    evaluation: the search itself held most of RL's `max_branches`, so the settling batch
    was rejected outright and 31% of searches - the ones with the most leaves, i.e. the
    ones the mode exists for - kept their unsettled scores.
    """

    instance_type = "whole_run"
    max_emulate_actions_items = 3  # doubles as the active-branch capacity

    def __init__(self) -> None:
        super().__init__()
        self.active: set[str] = set()
        self.rejected = 0

    async def emulate_actions(self, instance_id, items, *, timeout_s, simulation_options=None):
        if len(self.active) + len(items) > self.max_emulate_actions_items:
            self.rejected += 1
            raise RequestRejectedError(
                {"status": "rejected", "fault_kind": "active_branch_capacity"}
            )
        response = await super().emulate_actions(
            instance_id, items, timeout_s=timeout_s, simulation_options=simulation_options
        )
        self.active.update(item["branch_id"] for item in items)
        return response

    async def release_branches(self, instance_id, branch_ids, *, timeout_s):
        self.active.difference_update(branch_ids)
        return await super().release_branches(instance_id, branch_ids, timeout_s=timeout_s)


class ActiveBranchCapacityTest(unittest.IsolatedAsyncioTestCase):
    async def test_settling_frees_branches_before_asking_for_more(self) -> None:
        client = _CappedWholeRunClient()

        result = await _engine(client, boundary=True).search(
            "inst", _ROOT_DECISION, timeout_s=5.0
        )

        self.assertGreater(result.stats.leaves_turn_aligned, 0)
        self.assertEqual(result.best_root_action_id, "end")
        self.assertEqual(result.best_node_score, 20.0)


class ConfigTest(unittest.TestCase):
    def test_the_two_alignment_rules_are_mutually_exclusive(self) -> None:
        with self.assertRaises(ValueError):
            BeamSearchConfig(turn_aligned_leaves=True, turn_boundary_scoring=True)


if __name__ == "__main__":
    unittest.main()


_LINE_UNSETTLED = {"": 0.0, "big": 1.0, "small": 2.0, "small,small": 3.0}
_LINE_SETTLED = {"": 20.0, "big": 50.0, "small": 8.0, "small,small": 10.0}


class _ReleaseTrackingClient:
    """A Whole Run server that forgets released branches, like the real one.

    Two cards of different cost make the lines run out of energy at different plies, so
    leaves are born in different plies. The server rejects a whole batch that names a
    branch it no longer has - which is why settling once at the end lost even the leaves
    that were still alive.

    Unsettled, the two-small line looks best; settled, the big line wins.
    """

    instance_type = "whole_run"
    max_emulate_actions_items = 16
    pending_retry = None
    session_invalid = False

    def __init__(self) -> None:
        self.emulate_calls = []
        self.released = set()
        self.rejected_batches = 0
        self.line_by_branch = {"root": ("", 2)}

    def _dto(self, line, energy, *, settled):
        actions = [dict(_END)]
        if energy >= 2:
            actions.append(
                {"action_id": "big", "action_type": "card", "label": "BIG",
                 "is_available": True, "parameters": {"cardId": "BIG", "cost": 2}}
            )
        if energy >= 1:
            actions.append(
                {"action_id": "small", "action_type": "card", "label": "SMALL",
                 "is_available": True, "parameters": {"cardId": "SMALL", "cost": 1}}
            )
        table = _LINE_SETTLED if settled else _LINE_UNSETTLED
        return {
            "boundary": "stable",
            "turnNumber": 2 if settled else 1,
            "combatRoundNumber": 2 if settled else 1,
            "score": table[line],
            "legal_actions": [] if settled else actions,
        }

    async def emulate_actions(self, instance_id, items, *, timeout_s, simulation_options=None):
        del instance_id, timeout_s, simulation_options
        for item in items:
            if item["parent_branch_id"] in self.released:
                self.rejected_batches += 1
                raise RequestRejectedError({"status": "rejected", "fault_kind": "unknown_branch"})
        self.emulate_calls.append([dict(item) for item in items])
        results = {}
        for item in items:
            line, energy = self.line_by_branch[item["parent_branch_id"]]
            if item["action_id"] == "end":
                dto = self._dto(line, energy, settled=True)
                child = (line, energy)
            else:
                child = (
                    f"{line},{item['action_id']}" if line else item["action_id"],
                    energy - (2 if item["action_id"] == "big" else 1),
                )
                dto = self._dto(child[0], child[1], settled=False)
            self.line_by_branch[item["branch_id"]] = child
            results[item["branch_id"]] = {
                "status": "completed",
                "branch_id": item["branch_id"],
                "parent_branch_id": item["parent_branch_id"],
                "rng_id": item["rng_id"],
                "decision_point_id": f"d-{item['branch_id']}",
                "branch_log": [],
                "masked_emulator_dto": dto,
            }
        return {"status": "completed", "branch_results": results}

    async def cancel_branches(self, instance_id, branch_ids, *, timeout_s):
        del instance_id, timeout_s
        self.released.update(branch_ids)
        return {"status": "completed", "branch_statuses": {b: "cancelled" for b in branch_ids}}

    async def release_branches(self, instance_id, branch_ids, *, timeout_s):
        del instance_id, timeout_s
        self.released.update(branch_ids)
        return {"status": "completed", "branch_statuses": {b: "released" for b in branch_ids}}


_MULTI_PLY_ROOT = {
    "decision_point_id": "d-root",
    "masked_emulator_dto": {
        "boundary": "stable",
        "turnNumber": 1,
        "combatRoundNumber": 1,
        "score": 0.0,
        "legal_actions": [
            dict(_END),
            {"action_id": "big", "action_type": "card", "label": "BIG",
             "is_available": True, "parameters": {"cardId": "BIG", "cost": 2}},
            {"action_id": "small", "action_type": "card", "label": "SMALL",
             "is_available": True, "parameters": {"cardId": "SMALL", "cost": 1}},
        ],
    },
}


class ReleasedLeafTest(unittest.IsolatedAsyncioTestCase):
    async def test_a_leaf_is_settled_before_its_branch_is_released(self) -> None:
        """The big line becomes a leaf a ply before the small line does.

        Settling once at the end named that earlier leaf after it had been released, and
        the server rejected the whole batch - so no leaf was settled at all and the search
        answered with the unsettled order.
        """
        client = _ReleaseTrackingClient()
        engine = BeamSearchEngine(
            client,
            policy=_Policy(),
            value_fn=_Value(),
            config=BeamSearchConfig(
                max_depth=3,
                beam_width=4,
                top_k_actions=3,
                beam_searchable_action_types=COMBAT_BEAM_ACTION_TYPES,
                turn_boundary_scoring=True,
            ),
        )

        result = await engine.search("inst", _MULTI_PLY_ROOT, timeout_s=5.0)

        self.assertEqual(client.rejected_batches, 0)
        self.assertEqual(result.best_root_action_id, "big")
        self.assertEqual(result.best_node_score, 50.0)
