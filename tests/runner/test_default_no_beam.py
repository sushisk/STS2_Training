from __future__ import annotations

import unittest

from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.decision.search_modes import DEFAULT_SEARCH_MODE, search_mode_uses_beam
from sts2_training.runner.episode import EpisodeRunner, build_engine


class _NoEmulationClient:
    async def get_decision(self, instance_id: str, branch_id: str, *, timeout_s: float) -> dict:
        return {
            "decision_point_id": "d1",
            "masked_emulator_dto": {
                "legal_actions": [
                    {"action_id": "a", "action_type": "system", "is_available": True},
                    {"action_id": "b", "action_type": "system", "is_available": True},
                ]
            },
        }

    # Intentionally no emulate_actions method. The test would fail immediately if the
    # default engine attempted Beam Search instead of going straight to the fallback.


class DefaultNoBeamModeTest(unittest.IsolatedAsyncioTestCase):
    async def test_default_runner_engine_does_not_execute_beam_search(self) -> None:
        engine = build_engine(_NoEmulationClient())

        self.assertEqual(DEFAULT_SEARCH_MODE, "none")
        self.assertFalse(engine.beam_search_enabled)
        outcome = await engine.decide("inst-001", timeout_s=1.0)
        self.assertEqual(outcome.source, "heuristic_fallback")
        self.assertIn(outcome.chosen_action_id, {"a", "b"})
        self.assertIsNone(outcome.beam_result)

    async def test_episode_runner_constructor_uses_no_beam_default(self) -> None:
        runner = EpisodeRunner(_NoEmulationClient())
        self.assertFalse(runner._engine.beam_search_enabled)  # noqa: SLF001

    async def test_explicit_named_mode_reenables_beam(self) -> None:
        engine = build_engine(_NoEmulationClient(), search_mode="standard")
        self.assertTrue(engine.beam_search_enabled)

    async def test_explicit_config_reenables_beam(self) -> None:
        engine = build_engine(_NoEmulationClient(), search_mode=BeamSearchConfig(max_depth=1))
        self.assertTrue(engine.beam_search_enabled)

    def test_search_mode_toggle_contract(self) -> None:
        self.assertFalse(search_mode_uses_beam())
        self.assertFalse(search_mode_uses_beam("none"))
        for mode in ("shallow", "standard", "deep", "wide"):
            with self.subTest(mode=mode):
                self.assertTrue(search_mode_uses_beam(mode))


if __name__ == "__main__":
    unittest.main()
