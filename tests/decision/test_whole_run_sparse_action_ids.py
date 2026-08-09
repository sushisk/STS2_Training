from __future__ import annotations

import unittest

from sts2_training.decision.engine import CombatDecisionEngine


class _ChooseSecond:
    def select(self, actions):
        return actions[1]


class _FakeClient:
    def __init__(self, *, instance_type: str, emulate_supported: bool) -> None:
        self.instance_type = instance_type
        self.max_emulate_actions_items = 64 if emulate_supported else None
        self.committed_action_id: str | None = None

    async def get_decision(self, instance_id: str, branch_id: str, *, timeout_s: float):
        return {
            "status": "completed",
            "instance_id": instance_id,
            "branch_id": branch_id,
            "decision_point_id": "d-1",
            "masked_emulator_dto": {
                "legal_actions": [
                    {
                        "action_id": "0",
                        "action_type": "choice_reward_card",
                        "is_available": True,
                    },
                    {
                        "action_id": "3",
                        "action_type": "choice_reward_skip",
                        "is_available": True,
                    },
                ]
            },
        }

    async def commit_action(
        self,
        instance_id: str,
        decision_point_id: str,
        action_id: str,
        *,
        timeout_s: float,
    ):
        self.committed_action_id = action_id
        return {
            "status": "completed",
            "instance_id": instance_id,
            "branch_id": "root",
            "decision_point_id": "d-2",
            "masked_emulator_dto": {
                "run_terminal": True,
                "outcome": "victory",
                "legal_actions": [],
            },
        }


class WholeRunSparseActionIdTest(unittest.IsolatedAsyncioTestCase):
    async def test_whole_run_sparse_public_id_is_committed_by_position(self) -> None:
        client = _FakeClient(instance_type="whole_run", emulate_supported=False)
        engine = CombatDecisionEngine(client, fallback_selector=_ChooseSecond())

        await engine.decide_and_commit("inst-001", timeout_s=1.0)

        # STS2_RL WholeRunInstance currently resolves the wire token as an index into
        # legal_actions_raw. Public action_id "3" is the second item, so send "1".
        self.assertEqual(client.committed_action_id, "1")

    async def test_combat_keeps_public_action_id_unchanged(self) -> None:
        client = _FakeClient(instance_type="combat", emulate_supported=True)
        engine = CombatDecisionEngine(client, fallback_selector=_ChooseSecond())

        await engine.decide_and_commit("inst-001", timeout_s=1.0)

        self.assertEqual(client.committed_action_id, "3")

    async def test_missing_emulate_capability_alone_does_not_enable_ordinal_commit(self) -> None:
        client = _FakeClient(instance_type="future_mode", emulate_supported=False)
        engine = CombatDecisionEngine(client, fallback_selector=_ChooseSecond())

        await engine.decide_and_commit("inst-001", timeout_s=1.0)

        self.assertEqual(client.committed_action_id, "3")


if __name__ == "__main__":
    unittest.main()
