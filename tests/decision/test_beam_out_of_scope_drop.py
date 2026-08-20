"""Regression: continuation branches must be admitted, and never dropped in silence.

The Emulator publishes a `choice_target` continuation for `TargetType.AnyEnemy` cards
exactly when two or more enemies are alive. A `BeamSearchConfig` whose
`beam_searchable_action_types` omits the continuation types therefore removes every
targeted attack from the search in multi-enemy fights while leaving self-targeted cards
untouched - and does so without faulting a single branch, so every existing fault counter
stays at zero.

These tests pin both halves of that: the full Combat scope keeps the targeted branch, and
the narrow scope is at least *loud* about what it discards.
"""

from __future__ import annotations

import unittest

from sts2_training.decision.beam_search import BeamSearchConfig, BeamSearchEngine
from sts2_training.decision.combat_decision import COMBAT_BEAM_ACTION_TYPES
from sts2_training.decision.policy import ActionCandidate, PolicyModel
from sts2_training.decision.search_trace import (
    BranchFaultTrace,
    InMemorySearchTraceCollector,
    OutOfScopeDropTrace,
    SearchTraceEnd,
)
from sts2_training.decision.value import ValueModel

_ROOT_DECISION = {
    "decision_point_id": "root-decision",
    "masked_emulator_dto": {
        "boundary": "stable",
        "score": 0.0,
        "legal_actions": [
            # STRIKE against two living enemies: the Emulator answers with a pending
            # target selection rather than a settled Combat state.
            {"action_id": "strike", "action_type": "card", "is_available": True},
            {"action_id": "defend", "action_type": "card", "is_available": True},
        ],
    },
}


class _Policy(PolicyModel):
    def propose(self, legal_actions, masked_emulator_dto, *, top_k):
        del masked_emulator_dto
        return [ActionCandidate(action_id=action["action_id"]) for action in legal_actions[:top_k]]


class _Value(ValueModel):
    def evaluate(self, masked_emulator_dto):
        return float(masked_emulator_dto["score"])


class _TargetedCardClient:
    """Whole Run client where `strike` opens a target choice and `defend` resolves."""

    instance_type = "whole_run"
    max_emulate_actions_items = 8
    pending_retry = None
    session_invalid = False

    def __init__(self) -> None:
        self.emulate_calls: list[list[dict]] = []
        self.cancel_calls: list[list[str]] = []
        self.release_calls: list[list[str]] = []

    def _dto_for(self, action_id: str) -> dict:
        if action_id == "strike":
            return {
                "boundary": "pending_choice",
                "score": 0.0,
                "legal_actions": [
                    {
                        "action_id": "enemy-0",
                        "action_type": "choice_target",
                        "is_available": True,
                    },
                    {
                        "action_id": "enemy-1",
                        "action_type": "choice_target",
                        "is_available": True,
                    },
                ],
            }
        if action_id in ("enemy-0", "enemy-1"):
            return {"boundary": "stable", "terminal": True, "score": 10.0, "legal_actions": []}
        return {"boundary": "stable", "terminal": True, "score": 1.0, "legal_actions": []}

    async def emulate_actions(self, instance_id, items, *, timeout_s, simulation_options=None):
        del instance_id, timeout_s, simulation_options
        self.emulate_calls.append([dict(item) for item in items])
        branch_results = {}
        for item in items:
            branch_results[item["branch_id"]] = {
                "status": "completed",
                "branch_id": item["branch_id"],
                "parent_branch_id": item["parent_branch_id"],
                "rng_id": item["rng_id"],
                "decision_point_id": f"next-{item['branch_id']}",
                "branch_log": [],
                "masked_emulator_dto": self._dto_for(item["action_id"]),
            }
        return {"status": "completed", "branch_results": branch_results}

    async def cancel_branches(self, instance_id, branch_ids, *, timeout_s):
        del instance_id, timeout_s
        self.cancel_calls.append(list(branch_ids))
        return {
            "status": "completed",
            "branch_statuses": {branch_id: "cancelled" for branch_id in branch_ids},
        }

    async def release_branches(self, instance_id, branch_ids, *, timeout_s):
        del instance_id, timeout_s
        self.release_calls.append(list(branch_ids))
        return {
            "status": "completed",
            "branch_statuses": {branch_id: "released" for branch_id in branch_ids},
        }


def _engine(client, action_types, collector=None) -> BeamSearchEngine:
    return BeamSearchEngine(
        client,
        policy=_Policy(),
        value_fn=_Value(),
        config=BeamSearchConfig(
            max_depth=1,
            beam_width=4,
            top_k_actions=4,
            beam_searchable_action_types=frozenset(action_types),
        ),
        trace_collector=collector,
    )


class TargetedCardScopeTest(unittest.IsolatedAsyncioTestCase):
    async def test_full_combat_scope_keeps_the_targeted_card(self) -> None:
        client = _TargetedCardClient()
        engine = _engine(client, COMBAT_BEAM_ACTION_TYPES)

        result = await engine.search("inst-whole-run", _ROOT_DECISION, timeout_s=5.0)

        self.assertEqual(result.best_root_action_id, "strike")
        self.assertEqual(result.best_node_score, 10.0)
        self.assertEqual(result.stats.branches_out_of_scope, 0)
        self.assertEqual(result.stats.branches_faulted, 0)

    async def test_narrow_scope_drops_the_targeted_card(self) -> None:
        client = _TargetedCardClient()
        engine = _engine(client, {"system", "card", "potion"})

        result = await engine.search("inst-whole-run", _ROOT_DECISION, timeout_s=5.0)

        # The search still returns a plausible-looking answer; only the counter reveals
        # that attacking was never actually on the table.
        self.assertEqual(result.best_root_action_id, "defend")
        self.assertEqual(result.stats.branches_out_of_scope, 1)
        self.assertEqual(result.stats.branches_faulted, 0)

    async def test_narrow_scope_drop_is_traced_with_the_admission_evidence(self) -> None:
        client = _TargetedCardClient()
        collector = InMemorySearchTraceCollector()
        engine = _engine(client, {"system", "card", "potion"}, collector)

        await engine.search("inst-whole-run", _ROOT_DECISION, timeout_s=5.0)

        drops = [e for e in collector.events if isinstance(e, OutOfScopeDropTrace)]
        self.assertEqual(len(drops), 1)
        drop = drops[0]
        self.assertEqual(drop.action_id, "strike")
        self.assertEqual(drop.root_action_id, "strike")
        self.assertEqual(drop.action_type, "card")
        self.assertEqual(drop.boundary, "pending_choice")
        self.assertEqual(drop.observed_action_types, ("choice_target",))
        self.assertEqual(drop.allowed_action_types, ("card", "potion", "system"))

        # An out-of-scope drop is a configuration signal, not a branch fault.
        self.assertEqual([e for e in collector.events if isinstance(e, BranchFaultTrace)], [])
        ends = [e for e in collector.events if isinstance(e, SearchTraceEnd)]
        self.assertEqual([end.branches_out_of_scope for end in ends], [1])
        self.assertEqual([end.branches_faulted for end in ends], [0])

    async def test_full_combat_scope_records_no_drop_trace(self) -> None:
        client = _TargetedCardClient()
        collector = InMemorySearchTraceCollector()
        engine = _engine(client, COMBAT_BEAM_ACTION_TYPES, collector)

        await engine.search("inst-whole-run", _ROOT_DECISION, timeout_s=5.0)

        self.assertEqual([e for e in collector.events if isinstance(e, OutOfScopeDropTrace)], [])


if __name__ == "__main__":
    unittest.main()
