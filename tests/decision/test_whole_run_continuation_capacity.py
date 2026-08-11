from __future__ import annotations

import unittest
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_training.decision.beam_search import BeamSearchConfig, BeamSearchEngine
from sts2_training.decision.policy import ActionCandidate, PolicyModel
from sts2_training.decision.value import ValueModel


class _Policy(PolicyModel):
    def propose_batch(
        self,
        requests: Sequence[tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]]],
        *,
        top_k: int,
    ) -> list[list[ActionCandidate]]:
        proposals: list[list[ActionCandidate]] = []
        for legal_actions, _ in requests:
            proposals.append(
                [
                    ActionCandidate(str(action["action_id"]))
                    for action in legal_actions
                    if isinstance(action, Mapping) and action.get("is_available") is not False
                ][:top_k]
            )
        return proposals


class _Value(ValueModel):
    def evaluate(self, masked_emulator_dto: Mapping[str, Any]) -> float:
        return float(masked_emulator_dto.get("score", 0.0))


class _CapacityContinuationClient:
    instance_type = "whole_run"
    # Whole Run publishes the configured Branch capacity in this field. The same
    # capacity is both the single-request item ceiling and the total live Branch budget.
    max_emulate_actions_items = 8
    pending_retry = None
    session_invalid = False

    def __init__(self) -> None:
        self.active_branch_ids: set[str] = set()
        self.emulate_calls: list[list[dict[str, Any]]] = []
        self.release_calls: list[list[str]] = []
        self.cancel_calls: list[list[str]] = []
        self.max_observed_active = 0
        self._stable_score = 0

    @staticmethod
    def _continuation_dto() -> dict[str, Any]:
        return {
            "boundary": "pending_choice",
            "legal_actions": [
                {"action_id": "continue", "action_type": "choice_confirm", "is_available": True},
                {"action_id": "settle", "action_type": "choice_skip", "is_available": True},
            ],
        }

    def _stable_dto(self) -> dict[str, Any]:
        self._stable_score += 1
        return {
            "boundary": "stable",
            "score": float(self._stable_score),
            "legal_actions": [
                {"action_id": "attack", "action_type": "card", "is_available": True},
                {"action_id": "end", "action_type": "system", "is_available": True},
            ],
        }

    async def emulate_actions(
        self,
        instance_id: str,
        items: Sequence[Mapping[str, Any]],
        *,
        timeout_s: float,
        simulation_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        del instance_id, timeout_s, simulation_options
        normalized = [dict(item) for item in items]
        self.emulate_calls.append(normalized)

        assert len(normalized) <= self.max_emulate_actions_items
        for item in normalized:
            parent = item["parent_branch_id"]
            if parent != "root":
                assert parent in self.active_branch_ids, f"parent {parent!r} was released too early"
        assert len(self.active_branch_ids) + len(normalized) <= self.max_emulate_actions_items, (
            "Training exceeded Whole Run active Branch capacity before emulate_actions"
        )

        branch_results: dict[str, dict[str, Any]] = {}
        for item in normalized:
            branch_id = str(item["branch_id"])
            self.active_branch_ids.add(branch_id)
            parent = str(item["parent_branch_id"])
            action_id = str(item["action_id"])
            if parent == "root" or action_id == "continue":
                dto = self._continuation_dto()
            else:
                dto = self._stable_dto()
            branch_results[branch_id] = {
                "status": "completed",
                "branch_id": branch_id,
                "parent_branch_id": parent,
                "rng_id": int(item["rng_id"]),
                "decision_point_id": f"next-{branch_id}",
                "branch_log": [],
                "masked_emulator_dto": dto,
            }

        self.max_observed_active = max(self.max_observed_active, len(self.active_branch_ids))
        return {"status": "completed", "branch_results": branch_results}

    async def release_branches(
        self,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        del instance_id, timeout_s
        normalized = list(branch_ids)
        self.release_calls.append(normalized)
        self.active_branch_ids.difference_update(normalized)
        return {
            "status": "completed",
            "branch_statuses": {branch_id: "released" for branch_id in normalized},
        }

    async def cancel_branches(
        self,
        instance_id: str,
        branch_ids: Sequence[str],
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        del instance_id, timeout_s
        normalized = list(branch_ids)
        self.cancel_calls.append(normalized)
        self.active_branch_ids.difference_update(normalized)
        return {
            "status": "completed",
            "branch_statuses": {branch_id: "cancelled" for branch_id in normalized},
        }


class WholeRunContinuationCapacityTest(unittest.IsolatedAsyncioTestCase):
    async def test_waiting_stable_is_pruned_before_next_continuation_batch(self) -> None:
        client = _CapacityContinuationClient()
        engine = BeamSearchEngine(
            client,
            policy=_Policy(),
            value_fn=_Value(),
            config=BeamSearchConfig(
                beam_width=2,
                top_k_actions=2,
                max_depth=2,
                max_batch_size=8,
                max_continuation_steps=4,
                beam_searchable_action_types=frozenset(
                    {"card", "system", "choice_confirm", "choice_skip"}
                ),
            ),
        )
        root = {
            "decision_point_id": "root-decision",
            "masked_emulator_dto": {
                "boundary": "stable",
                "legal_actions": [
                    {"action_id": "root-a", "action_type": "card", "is_available": True},
                    {"action_id": "root-b", "action_type": "system", "is_available": True},
                ],
            },
        }

        result = await engine.search("inst", root, timeout_s=2.0)

        self.assertIsNotNone(result.best_root_action_id)
        # Root + three continuation rounds are enough for the old implementation to hit
        # 6 live Branches and then try to add four more against an 8-Branch cap.
        self.assertGreaterEqual(len(client.emulate_calls), 4)
        self.assertLessEqual(client.max_observed_active, client.max_emulate_actions_items)
        self.assertTrue(client.release_calls)
        self.assertEqual(client.active_branch_ids, set())


if __name__ == "__main__":
    unittest.main()
