from __future__ import annotations

import unittest

from sts2_training.decision.engine import CombatDecisionEngine
from sts2_training.runner.episode import EpisodeRunner

_ACTION = {"action_id": "0", "action_type": "system", "is_available": True}


def _decision(decision_point_id: str, *, terminal: bool = False) -> dict:
    dto = {"legal_actions": [] if terminal else [_ACTION]}
    if terminal:
        dto.update({"terminal": True, "outcome": "victory"})
    return {
        "status": "completed",
        "branch_id": "root",
        "decision_point_id": decision_point_id,
        "masked_emulator_dto": dto,
    }


class _FakeClient:
    max_emulate_actions_items = 64
    pending_retry = None
    session_invalid = False

    def __init__(self) -> None:
        self._decisions = [_decision("d0"), _decision("d1"), _decision("d2", terminal=True)]
        self._index = 0
        self.get_decision_calls = 0
        self.commit_action_calls = 0
        self.close_instance_calls = 0

    async def get_decision(self, instance_id: str, branch_id: str, *, timeout_s: float):
        self.get_decision_calls += 1
        return {"instance_id": instance_id, **self._decisions[self._index]}

    async def commit_action(
        self,
        instance_id: str,
        decision_point_id: str,
        action_id: str,
        *,
        timeout_s: float,
    ):
        self.commit_action_calls += 1
        self._index += 1
        return {"instance_id": instance_id, **self._decisions[self._index]}

    async def close_instance(self, instance_id: str, *, timeout_s: float):
        self.close_instance_calls += 1
        return {"status": "completed", "instance_id": instance_id}


class ReuseCommitDecisionTest(unittest.IsolatedAsyncioTestCase):
    async def test_episode_fetches_only_first_decision(self) -> None:
        client = _FakeClient()
        runner = EpisodeRunner(client, CombatDecisionEngine(client))

        result = await runner.run("inst-001", decision_timeout_s=1.0)

        self.assertEqual(result.decisions_made, 2)
        self.assertEqual(result.final_dto["outcome"], "victory")
        self.assertEqual(client.commit_action_calls, 2)
        self.assertEqual(client.get_decision_calls, 1)
        self.assertEqual(client.close_instance_calls, 1)


if __name__ == "__main__":
    unittest.main()
