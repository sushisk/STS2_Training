from __future__ import annotations

import asyncio
import unittest

from sts2_training.decision.beam_search import BeamSearchConfig, BeamSearchEngine
from sts2_training.decision.combat_decision import COMBAT_BEAM_ACTION_TYPES
from sts2_training.decision.policy import PriorHeuristicPolicy
from sts2_training.decision.value import ValueModel


def _action(action_id: str, action_type: str) -> dict:
    return {"action_id": action_id, "action_type": action_type, "is_available": True}


def _dto(actions: list[dict], *, score: float = 0.0) -> dict:
    return {"score": score, "legal_actions": actions}


class _ScoreValue(ValueModel):
    def evaluate(self, dto):
        return float(dto.get("score", 0.0))


class _DelayedScriptedClient:
    instance_type = "combat"
    max_emulate_actions_items = 64
    pending_retry = None
    session_invalid = False

    def __init__(self, transitions: dict[str, dict], *, delay_s: float) -> None:
        self.transitions = transitions
        self.delay_s = delay_s
        self.emulate_calls = 0

    async def emulate_actions(self, instance_id, items, *, timeout_s, simulation_options=None):
        self.emulate_calls += 1
        await asyncio.sleep(self.delay_s)
        results = {}
        for item in items:
            dto = self.transitions[item["action_id"]]
            results[item["branch_id"]] = {
                "status": "completed",
                "branch_id": item["branch_id"],
                "parent_branch_id": item["parent_branch_id"],
                "rng_id": item["rng_id"],
                "decision_point_id": f"d-{item['branch_id']}",
                "branch_log": [],
                "masked_emulator_dto": dto,
            }
        return {"branch_results": results}


def _engine(client: _DelayedScriptedClient) -> BeamSearchEngine:
    return BeamSearchEngine(
        client,
        policy=PriorHeuristicPolicy(),
        value_fn=_ScoreValue(),
        config=BeamSearchConfig(
            max_depth=1,
            top_k_actions=4,
            beam_width=8,
            time_budget_ms=50.0,
            release_branches_on_finish=False,
            beam_searchable_action_types=COMBAT_BEAM_ACTION_TYPES,
        ),
    )


class BeamSearchTimeBudgetTest(unittest.IsolatedAsyncioTestCase):
    async def test_time_budget_drops_unresolved_continuation_candidate(self) -> None:
        client = _DelayedScriptedClient(
            {
                "play-card": _dto([_action("pick", "choice_card")]),
            },
            delay_s=0.075,
        )
        engine = _engine(client)
        root = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": _dto([_action("play-card", "card")]),
        }

        result = await engine.search("inst-001", root, timeout_s=5.0)

        self.assertEqual(result.reason, "time_budget")
        self.assertIsNone(result.best_root_action_id)
        self.assertIsNone(result.best_node)
        self.assertEqual(client.emulate_calls, 1)

    async def test_time_budget_keeps_resolved_stable_sibling_only(self) -> None:
        client = _DelayedScriptedClient(
            {
                "play-card": _dto([_action("pick", "choice_card")]),
                "end": _dto([_action("next", "system")], score=10.0),
            },
            delay_s=0.075,
        )
        engine = _engine(client)
        root = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": _dto(
                [
                    _action("play-card", "card"),
                    _action("end", "system"),
                ]
            ),
        }

        result = await engine.search("inst-001", root, timeout_s=5.0)

        self.assertEqual(result.reason, "time_budget")
        self.assertEqual(result.best_root_action_id, "end")
        self.assertIsNotNone(result.best_node)
        self.assertEqual(result.best_node.value, 10.0)
        self.assertEqual(client.emulate_calls, 1)


if __name__ == "__main__":
    unittest.main()
