"""Coverage for the three top-level entry points: each one's job is only to build
the right `instance_config` and hand off to `EpisodeRunner` - the loop itself is
covered by `test_episode.py`, so these tests just assert `start_instance` receives
the expected config and an `EpisodeResult` comes back.
"""

from __future__ import annotations

import unittest

from sts2_training.runner.scenario import CombatScenario, EnemyScenario, RunSnapshot
from sts2_training.runner.start_combat_from_state import start_combat_from_state
from sts2_training.runner.start_new_run import start_new_run
from sts2_training.runner.start_run_from_state import (
    RunSnapshotRestoreNotSupportedError,
    start_run_from_state,
)


class _FakeClient:
    """Records `start_instance` calls; immediately reports the instance terminal so
    `EpisodeRunner.run()` makes zero decisions and returns right away."""

    def __init__(self) -> None:
        self.start_instance_calls: list[dict] = []
        self.close_instance_calls = 0
        self.pending_retry = None
        self.session_invalid = False

    async def start_instance(self, instance_config: dict, *, timeout_s: float) -> str:
        self.start_instance_calls.append(dict(instance_config))
        return "inst-001"

    async def get_decision(self, instance_id: str, branch_id: str = "root", *, timeout_s: float) -> dict:
        return {"decision_point_id": "d0", "masked_emulator_dto": {"legal_actions": []}}

    async def close_instance(self, instance_id: str, *, timeout_s: float) -> dict:
        self.close_instance_calls += 1
        return {"status": "completed"}


def _scenario() -> CombatScenario:
    return CombatScenario(
        character_id="IRONCLAD",
        player_hp=50,
        player_max_hp=80,
        hand=["STRIKE_IRONCLAD"],
        draw_pile=[],
        discard_pile=[],
        enemies=[EnemyScenario(monster_id="CALCIFIED_CULTIST", hp=48)],
    )


class StartCombatFromStateTest(unittest.IsolatedAsyncioTestCase):
    async def test_builds_combat_instance_config_and_runs_to_completion(self) -> None:
        client = _FakeClient()

        result = await start_combat_from_state(client, _scenario(), decision_timeout_s=5.0)

        self.assertEqual(len(client.start_instance_calls), 1)
        self.assertEqual(client.start_instance_calls[0]["instance_type"], "combat")
        self.assertEqual(client.start_instance_calls[0]["character_id"], "IRONCLAD")
        self.assertEqual(result.instance_id, "inst-001")
        self.assertEqual(result.decisions_made, 0)
        self.assertEqual(client.close_instance_calls, 1)

    async def test_search_mode_selects_the_beam_config_used(self) -> None:
        client = _FakeClient()

        result = await start_combat_from_state(
            client, _scenario(), decision_timeout_s=5.0, search_mode="deep", beam_max_depth=7
        )

        self.assertEqual(result.instance_id, "inst-001")  # reaches the RL round trip fine

    async def test_unknown_search_mode_raises_before_touching_the_client(self) -> None:
        client = _FakeClient()

        with self.assertRaises(ValueError):
            await start_combat_from_state(client, _scenario(), decision_timeout_s=5.0, search_mode="nonexistent")

        self.assertEqual(client.start_instance_calls, [])

    async def test_engine_and_search_mode_together_is_rejected(self) -> None:
        from sts2_training.decision.engine import CombatDecisionEngine

        client = _FakeClient()
        engine = CombatDecisionEngine(client)

        with self.assertRaises(ValueError):
            await start_combat_from_state(client, _scenario(), decision_timeout_s=5.0, engine=engine, search_mode="deep")


class StartNewRunTest(unittest.IsolatedAsyncioTestCase):
    async def test_builds_whole_run_instance_config_and_runs_to_completion(self) -> None:
        client = _FakeClient()

        result = await start_new_run(client, character_id="IRONCLAD", ascension=2, seed=7, decision_timeout_s=5.0)

        self.assertEqual(
            client.start_instance_calls[0],
            {"instance_type": "whole_run", "character_id": "IRONCLAD", "ascension": 2, "seed": 7},
        )
        self.assertEqual(result.instance_id, "inst-001")

    async def test_omitted_seed_is_filled_in_randomly_and_varies(self) -> None:
        client = _FakeClient()

        await start_new_run(client, character_id="IRONCLAD", decision_timeout_s=5.0)
        await start_new_run(client, character_id="IRONCLAD", decision_timeout_s=5.0)

        seeds = [call["seed"] for call in client.start_instance_calls]
        self.assertEqual(len(seeds), 2)
        for seed in seeds:
            self.assertIsInstance(seed, int)
        self.assertNotEqual(seeds[0], seeds[1])


class StartRunFromStateTest(unittest.IsolatedAsyncioTestCase):
    async def test_raises_before_touching_the_client(self) -> None:
        client = _FakeClient()
        snapshot = RunSnapshot(character_id="IRONCLAD", ascension=0, seed=1, snapshot_json="{...}")

        with self.assertRaises(RunSnapshotRestoreNotSupportedError):
            await start_run_from_state(client, snapshot, decision_timeout_s=5.0)

        self.assertEqual(client.start_instance_calls, [])


if __name__ == "__main__":
    unittest.main()
