"""Whole Run Combat Beam boundary and terminal-outcome regressions."""

from __future__ import annotations

import unittest

from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.decision.engine import CombatDecisionEngine
from sts2_training.decision.policy import ActionCandidate, PolicyModel
from sts2_training.decision.value import DEFAULT_WEIGHTS, HeuristicValueFunction, ValueModel


class _Policy(PolicyModel):
    def propose(self, legal_actions, masked_emulator_dto, *, top_k):
        del masked_emulator_dto
        return [ActionCandidate(action_id=action["action_id"]) for action in legal_actions[:top_k]]


class _Value(ValueModel):
    def evaluate(self, masked_emulator_dto):
        return float(masked_emulator_dto["score"])


class _Fallback:
    def select(self, legal_actions):
        return legal_actions[0]


class _WholeRunBeamClient:
    instance_type = "whole_run"
    max_emulate_actions_items = 8
    pending_retry = None
    session_invalid = False

    def __init__(self) -> None:
        self.emulate_calls: list[list[dict]] = []
        self.cancel_calls: list[list[str]] = []
        self.release_calls: list[list[str]] = []

    async def emulate_actions(self, instance_id, items, *, timeout_s, simulation_options=None):
        del instance_id, timeout_s, simulation_options
        self.emulate_calls.append([dict(item) for item in items])
        branch_results = {}
        for item in items:
            score = 10.0 if item["action_id"] == "good" else 1.0
            branch_results[item["branch_id"]] = {
                "status": "completed",
                "branch_id": item["branch_id"],
                "parent_branch_id": item["parent_branch_id"],
                "rng_id": item["rng_id"],
                "decision_point_id": f"next-{item['branch_id']}",
                "branch_log": [],
                "masked_emulator_dto": {
                    "terminal": True,
                    "score": score,
                    "legal_actions": [],
                },
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


class _WholeRunBoundaryExitClient(_WholeRunBeamClient):
    async def emulate_actions(self, instance_id, items, *, timeout_s, simulation_options=None):
        del instance_id, timeout_s, simulation_options
        self.emulate_calls.append([dict(item) for item in items])
        if len(self.emulate_calls) > 1:
            raise AssertionError("Beam must not expand a non-Combat Whole Run boundary")

        branch_results = {}
        for item in items:
            branch_results[item["branch_id"]] = {
                "status": "completed",
                "branch_id": item["branch_id"],
                "parent_branch_id": item["parent_branch_id"],
                "rng_id": item["rng_id"],
                "decision_point_id": f"event-{item['branch_id']}",
                "branch_log": [],
                "masked_emulator_dto": {
                    "boundary": "event_choice",
                    "legal_actions": [
                        {
                            "action_id": "confirm",
                            "action_type": "choice_confirm",
                            "is_available": True,
                        },
                        {
                            "action_id": "skip",
                            "action_type": "choice_skip",
                            "is_available": True,
                        },
                    ],
                },
            }
        return {"status": "completed", "branch_results": branch_results}


class _WholeRunCombatCompletionClient(_WholeRunBeamClient):
    async def emulate_actions(self, instance_id, items, *, timeout_s, simulation_options=None):
        del instance_id, timeout_s, simulation_options
        self.emulate_calls.append([dict(item) for item in items])
        if len(self.emulate_calls) > 1:
            raise AssertionError("combat completion must terminate the Beam branch")

        branch_results = {}
        for item in items:
            if item["action_id"] == "win":
                dto = {
                    # Whole Run settles past Combat before publishing the next boundary.
                    # `transition` is therefore the authoritative combat-end signal.
                    "boundary": "reward_select",
                    "transition": {"kind": "combat_completed", "victory": True},
                    "legal_actions": [
                        {
                            "action_id": "reward-card",
                            "action_type": "choice_reward_card",
                            "is_available": True,
                        }
                    ],
                }
            else:
                dto = {
                    "run_terminal": True,
                    "outcome": "defeat",
                    "legal_actions": [],
                }
            branch_results[item["branch_id"]] = {
                "status": "completed",
                "branch_id": item["branch_id"],
                "parent_branch_id": item["parent_branch_id"],
                "rng_id": item["rng_id"],
                "decision_point_id": f"next-{item['branch_id']}",
                "branch_log": [],
                "masked_emulator_dto": dto,
            }
        return {"status": "completed", "branch_results": branch_results}


class WholeRunCombatBeamTest(unittest.IsolatedAsyncioTestCase):
    async def test_capable_whole_run_uses_beam_search(self) -> None:
        client = _WholeRunBeamClient()
        engine = CombatDecisionEngine(
            client,
            policy=_Policy(),
            value_fn=_Value(),
            beam_config=BeamSearchConfig(
                max_depth=1,
                beam_width=2,
                top_k_actions=2,
                beam_searchable_action_types=frozenset({"card", "system"}),
            ),
        )
        decision = {
            "decision_point_id": "root-decision",
            "masked_emulator_dto": {
                "boundary": "stable",
                "legal_actions": [
                    {"action_id": "good", "action_type": "card", "is_available": True},
                    {"action_id": "bad", "action_type": "system", "is_available": True},
                ],
            },
        }

        outcome = await engine.decide(
            "inst-whole-run",
            timeout_s=2.0,
            decision=decision,
        )

        self.assertEqual(outcome.source, "beam_search")
        self.assertEqual(outcome.chosen_action_id, "good")
        self.assertEqual(len(client.emulate_calls), 1)
        self.assertEqual(len(client.emulate_calls[0]), 2)
        self.assertEqual(len(client.cancel_calls), 1)
        self.assertEqual(len(client.release_calls), 1)

    async def test_non_combat_whole_run_boundary_skips_beam_even_with_combat_like_actions(
        self,
    ) -> None:
        client = _WholeRunBeamClient()
        engine = CombatDecisionEngine(
            client,
            policy=_Policy(),
            value_fn=_Value(),
            beam_config=BeamSearchConfig(
                max_depth=1,
                beam_width=2,
                top_k_actions=2,
                beam_searchable_action_types=frozenset({"choice_confirm", "choice_skip"}),
            ),
            fallback_selector=_Fallback(),
        )
        decision = {
            "decision_point_id": "event-decision",
            "masked_emulator_dto": {
                "boundary": "event_choice",
                "legal_actions": [
                    {
                        "action_id": "confirm",
                        "action_type": "choice_confirm",
                        "is_available": True,
                    },
                    {
                        "action_id": "skip",
                        "action_type": "choice_skip",
                        "is_available": True,
                    },
                ],
            },
        }

        outcome = await engine.decide(
            "inst-whole-run",
            timeout_s=2.0,
            decision=decision,
        )

        self.assertEqual(outcome.source, "heuristic_fallback")
        self.assertEqual(outcome.chosen_action_id, "confirm")
        self.assertIsNotNone(outcome.beam_result)
        self.assertEqual(outcome.beam_result.reason, "not_beam_searchable")
        self.assertEqual(client.emulate_calls, [])

    async def test_whole_run_branch_leaving_combat_scope_is_not_expanded(self) -> None:
        client = _WholeRunBoundaryExitClient()
        engine = CombatDecisionEngine(
            client,
            policy=_Policy(),
            value_fn=_Value(),
            beam_config=BeamSearchConfig(
                max_depth=2,
                beam_width=2,
                top_k_actions=2,
                beam_searchable_action_types=frozenset(
                    {"card", "system", "choice_confirm", "choice_skip"}
                ),
            ),
            fallback_selector=_Fallback(),
        )
        decision = {
            "decision_point_id": "root-decision",
            "masked_emulator_dto": {
                "boundary": "stable",
                "legal_actions": [
                    {"action_id": "good", "action_type": "card", "is_available": True},
                    {"action_id": "bad", "action_type": "system", "is_available": True},
                ],
            },
        }

        outcome = await engine.decide(
            "inst-whole-run",
            timeout_s=2.0,
            decision=decision,
        )

        self.assertEqual(outcome.source, "heuristic_fallback")
        self.assertEqual(outcome.chosen_action_id, "good")
        self.assertIsNotNone(outcome.beam_result)
        self.assertEqual(outcome.beam_result.reason, "not_beam_searchable")
        self.assertEqual(len(client.emulate_calls), 1)
        self.assertEqual(len(client.cancel_calls), 1)
        self.assertEqual(len(client.release_calls), 1)

    async def test_whole_run_combat_completion_is_scored_as_terminal(self) -> None:
        client = _WholeRunCombatCompletionClient()
        engine = CombatDecisionEngine(
            client,
            policy=_Policy(),
            value_fn=HeuristicValueFunction(),
            beam_config=BeamSearchConfig(
                max_depth=2,
                beam_width=2,
                top_k_actions=2,
                beam_searchable_action_types=frozenset({"card", "system"}),
            ),
        )
        decision = {
            "decision_point_id": "root-decision",
            "masked_emulator_dto": {
                "boundary": "stable",
                "legal_actions": [
                    {"action_id": "win", "action_type": "card", "is_available": True},
                    {"action_id": "lose", "action_type": "system", "is_available": True},
                ],
            },
        }

        outcome = await engine.decide(
            "inst-whole-run",
            timeout_s=2.0,
            decision=decision,
        )

        self.assertEqual(outcome.source, "beam_search")
        self.assertEqual(outcome.chosen_action_id, "win")
        self.assertIsNotNone(outcome.beam_result)
        self.assertIsNotNone(outcome.beam_result.best_node)
        self.assertTrue(outcome.beam_result.best_node.terminal)
        self.assertEqual(outcome.beam_result.best_value, DEFAULT_WEIGHTS["victory_bonus"])
        self.assertEqual(len(client.emulate_calls), 1)
        self.assertEqual(len(client.cancel_calls), 1)
        self.assertEqual(len(client.release_calls), 1)

    def test_run_victory_receives_victory_bonus(self) -> None:
        value = HeuristicValueFunction().evaluate(
            {"run_terminal": True, "outcome": "run_victory"}
        )
        self.assertEqual(value, DEFAULT_WEIGHTS["victory_bonus"])


if __name__ == "__main__":
    unittest.main()
