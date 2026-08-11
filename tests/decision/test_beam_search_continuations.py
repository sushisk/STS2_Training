from __future__ import annotations

import unittest

from sts2_training.decision.beam_search import BeamSearchConfig, BeamSearchEngine
from sts2_training.decision.combat_decision import COMBAT_BEAM_ACTION_TYPES
from sts2_training.decision.policy import PriorHeuristicPolicy
from sts2_training.decision.value import ValueModel


def _action(action_id: str, action_type: str) -> dict:
    return {"action_id": action_id, "action_type": action_type, "is_available": True}


def _dto(actions: list[dict], *, score: float = 0.0, terminal: bool = False) -> dict:
    return {
        "terminal": terminal,
        "outcome": "victory" if terminal else None,
        "score": score,
        "legal_actions": actions,
    }


class _ScoreValue(ValueModel):
    def evaluate(self, dto):
        return float(dto.get("score", 0.0))


class _ScriptedClient:
    instance_type = "combat"
    max_emulate_actions_items = 64
    pending_retry = None
    session_invalid = False

    def __init__(self, transitions: dict[str, dict]) -> None:
        self.transitions = transitions
        self.emulate_calls: list[list[dict]] = []

    async def emulate_actions(self, instance_id, items, *, timeout_s, simulation_options=None):
        self.emulate_calls.append([dict(item) for item in items])
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

    async def cancel_branches(self, instance_id, branch_ids, *, timeout_s):
        return {"status": "completed"}

    async def release_branches(self, instance_id, branch_ids, *, timeout_s):
        return {"status": "completed"}


def _engine(
    client: _ScriptedClient,
    *,
    max_depth: int = 1,
    max_continuation_steps: int = 8,
) -> BeamSearchEngine:
    return BeamSearchEngine(
        client,
        policy=PriorHeuristicPolicy(),
        value_fn=_ScoreValue(),
        config=BeamSearchConfig(
            max_depth=max_depth,
            top_k_actions=4,
            beam_width=8,
            max_continuation_steps=max_continuation_steps,
            beam_searchable_action_types=COMBAT_BEAM_ACTION_TYPES,
        ),
    )


class BeamSearchContinuationTest(unittest.IsolatedAsyncioTestCase):
    async def test_card_target_continuation_does_not_consume_extra_combat_depth(self) -> None:
        client = _ScriptedClient(
            {
                "play-card": _dto([_action("target", "choice_target")]),
                "target": _dto([], score=20.0, terminal=True),
            }
        )
        engine = _engine(client, max_depth=1)
        root = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": _dto([_action("play-card", "card")]),
        }

        result = await engine.search("inst-001", root, timeout_s=5.0)

        self.assertEqual(result.best_root_action_id, "play-card")
        self.assertEqual(result.best_value, 20.0)
        self.assertEqual(result.best_node.combat_depth, 1)
        self.assertEqual(result.best_node.depth, 2)
        self.assertEqual(len(client.emulate_calls), 2)

    async def test_choice_card_then_confirm_resolves_after_combat_depth_limit(self) -> None:
        client = _ScriptedClient(
            {
                "drink": _dto([_action("pick", "choice_card")]),
                "pick": _dto([_action("confirm", "choice_confirm")]),
                "confirm": _dto([], score=25.0, terminal=True),
            }
        )
        engine = _engine(client, max_depth=1)
        root = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": _dto([_action("drink", "potion")]),
        }

        result = await engine.search("inst-001", root, timeout_s=5.0)

        self.assertEqual(result.best_root_action_id, "drink")
        self.assertEqual(result.best_value, 25.0)
        self.assertEqual(result.best_node.combat_depth, 1)
        self.assertEqual(result.best_node.depth, 3)
        self.assertEqual(len(client.emulate_calls), 3)

    async def test_variable_count_choice_can_take_multiple_cards_before_confirm(self) -> None:
        client = _ScriptedClient(
            {
                "play-card": _dto([_action("pick-1", "choice_card")]),
                "pick-1": _dto([_action("pick-2", "choice_card")]),
                "pick-2": _dto([_action("confirm", "choice_confirm")]),
                "confirm": _dto([], score=30.0, terminal=True),
            }
        )
        engine = _engine(client)
        root = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": _dto([_action("play-card", "card")]),
        }

        result = await engine.search("inst-001", root, timeout_s=5.0)

        self.assertEqual(result.best_root_action_id, "play-card")
        self.assertEqual(result.best_value, 30.0)
        self.assertEqual(result.best_node.combat_depth, 1)
        self.assertEqual(result.best_node.depth, 4)
        self.assertEqual(len(client.emulate_calls), 4)

    async def test_continuation_step_limit_safely_returns_no_unresolved_candidate(self) -> None:
        loop = [_action("loop", "choice_card")]
        client = _ScriptedClient({"loop": _dto(loop, score=1.0)})
        engine = _engine(client, max_continuation_steps=2)
        root = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": _dto(loop),
        }

        result = await engine.search("inst-001", root, timeout_s=5.0)

        self.assertEqual(result.reason, "max_continuation_steps")
        self.assertIsNone(result.best_root_action_id)
        self.assertIsNone(result.best_node)
        self.assertEqual(len(client.emulate_calls), 2)


if __name__ == "__main__":
    unittest.main()
