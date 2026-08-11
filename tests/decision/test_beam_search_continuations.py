from __future__ import annotations

import unittest

from sts2_training.decision.beam_search import BeamSearchConfig, BeamSearchEngine
from sts2_training.decision.policy import ActionCandidate


_CONTINUATION_AWARE_TYPES = frozenset(
    {
        "system",
        "card",
        "potion",
        "choice_target",
        "choice_card",
        "choice_confirm",
        "choice_skip",
    }
)


def _action(action_id: str, action_type: str) -> dict:
    return {"action_id": action_id, "action_type": action_type, "is_available": True}


def _dto(actions: list[dict], *, score: float = 0.0, terminal: bool = False) -> dict:
    return {
        "legal_actions": actions,
        "score": score,
        "terminal": terminal,
        **({"outcome": "victory"} if terminal else {}),
    }


class _AllActionsPolicy:
    def propose_batch(self, requests, *, top_k: int):
        return [
            [ActionCandidate(action["action_id"]) for action in actions[:top_k]]
            for actions, _dto_value in requests
        ]


class _ScoreValue:
    def evaluate_batch(self, dtos):
        return [float(dto.get("score", 0.0)) for dto in dtos]


class _ScriptedClient:
    instance_type = "combat"
    max_emulate_actions_items = 64
    pending_retry = None
    session_invalid = False

    def __init__(self, results_by_action_id: dict[str, dict]) -> None:
        self.results_by_action_id = results_by_action_id
        self.emulate_calls: list[list[dict]] = []

    async def emulate_actions(
        self,
        instance_id: str,
        items,
        *,
        timeout_s: float,
        simulation_options=None,
    ) -> dict:
        copied_items = [dict(item) for item in items]
        self.emulate_calls.append(copied_items)
        branch_results = {}
        for item in copied_items:
            dto = self.results_by_action_id[item["action_id"]]
            branch_results[item["branch_id"]] = {
                "branch_id": item["branch_id"],
                "parent_branch_id": item["parent_branch_id"],
                "rng_id": item["rng_id"],
                "status": "completed",
                "decision_point_id": f"d-{item['branch_id']}",
                "branch_log": [],
                "masked_emulator_dto": dto,
            }
        return {"branch_results": branch_results}

    async def cancel_branches(self, instance_id: str, branch_ids, *, timeout_s: float) -> None:
        return None

    async def release_branches(self, instance_id: str, branch_ids, *, timeout_s: float) -> None:
        return None


def _engine(client: _ScriptedClient, *, max_continuation_steps: int = 8) -> BeamSearchEngine:
    return BeamSearchEngine(
        client,
        policy=_AllActionsPolicy(),
        value_fn=_ScoreValue(),
        config=BeamSearchConfig(
            max_depth=1,
            max_continuation_steps=max_continuation_steps,
            beam_width=8,
            top_k_actions=4,
            beam_searchable_action_types=_CONTINUATION_AWARE_TYPES,
        ),
    )


class BeamSearchContinuationTest(unittest.IsolatedAsyncioTestCase):
    async def test_card_then_target_is_explored_after_combat_depth_is_spent(self) -> None:
        target_actions = [
            _action("target-bad", "choice_target"),
            _action("target-good", "choice_target"),
        ]
        client = _ScriptedClient(
            {
                "play-card": _dto(target_actions),
                "target-bad": _dto([], score=-10.0, terminal=True),
                "target-good": _dto([], score=10.0, terminal=True),
            }
        )
        engine = _engine(client)
        root = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": _dto([_action("play-card", "card")]),
        }

        result = await engine.search("inst-001", root, timeout_s=5.0)

        self.assertEqual(result.best_root_action_id, "play-card")
        self.assertEqual(result.best_value, 10.0)
        self.assertEqual(result.best_node.combat_depth, 1)
        self.assertEqual(result.best_node.depth, 2)
        self.assertEqual(len(client.emulate_calls), 2)

    async def test_potion_choice_card_then_confirm_completes_after_depth_limit(self) -> None:
        client = _ScriptedClient(
            {
                "use-potion": _dto([_action("pick-card", "choice_card")]),
                "pick-card": _dto([_action("confirm", "choice_confirm")]),
                "confirm": _dto([], score=20.0, terminal=True),
            }
        )
        engine = _engine(client)
        root = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": _dto([_action("use-potion", "potion")]),
        }

        result = await engine.search("inst-001", root, timeout_s=5.0)

        self.assertEqual(result.best_root_action_id, "use-potion")
        self.assertEqual(result.best_value, 20.0)
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

    async def test_continuation_step_limit_stops_pathological_choice_loop(self) -> None:
        loop = [_action("loop", "choice_card")]
        client = _ScriptedClient({"loop": _dto(loop, score=1.0)})
        engine = _engine(client, max_continuation_steps=2)
        root = {
            "decision_point_id": "d-root",
            "masked_emulator_dto": _dto(loop),
        }

        result = await engine.search("inst-001", root, timeout_s=5.0)

        self.assertEqual(result.reason, "max_continuation_steps")
        self.assertEqual(result.best_node.depth, 2)
        self.assertEqual(result.best_node.combat_depth, 0)
        self.assertEqual(len(client.emulate_calls), 2)


if __name__ == "__main__":
    unittest.main()
