"""Every runner entry point must build engines with the full Combat Beam scope.

`resolve_search_mode()` returns budget-shaped configs carrying the conservative
`BeamSearchConfig` default (`{"system", "card", "potion"}`), and `CombatDecisionEngine`
treats an explicitly supplied `beam_config` as authoritative. A runner that forgets to
widen the scope therefore drops every continuation branch - i.e. every targeted card in a
multi-enemy fight - with no fault and no error. `runner/beam_scope.py` exists so that
widening happens in exactly one place; these tests pin each entry point to it.
"""

from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.decision.combat_decision import COMBAT_BEAM_ACTION_TYPES
from sts2_training.runner import episode as episode_module
from sts2_training.runner import floor_reach_eval as floor_reach_eval_module
from sts2_training.runner import self_play as self_play_module
from sts2_training.runner import stable_pruner_ab as stable_pruner_ab_module
from sts2_training.runner.beam_scope import runner_combat_beam_config


class _Client:
    instance_type = "whole_run"


class RunnerBeamScopeTest(unittest.TestCase):
    def test_helper_widens_scope_and_preserves_the_budget(self) -> None:
        narrow = BeamSearchConfig(max_depth=3, beam_width=16, top_k_actions=6)
        self.assertNotIn("choice_target", narrow.beam_searchable_action_types)

        widened = runner_combat_beam_config(narrow)

        self.assertEqual(widened.beam_searchable_action_types, COMBAT_BEAM_ACTION_TYPES)
        self.assertEqual(widened.max_depth, 3)
        self.assertEqual(widened.beam_width, 16)
        self.assertEqual(widened.top_k_actions, 6)
        self.assertIsNot(widened.simulation_options, narrow.simulation_options)

    def test_build_engine_uses_the_full_combat_scope(self) -> None:
        engine = episode_module.build_engine(_Client(), search_mode="standard")

        self.assertEqual(
            engine.beam_search.config.beam_searchable_action_types,
            COMBAT_BEAM_ACTION_TYPES,
        )

    def test_floor_reach_eval_engine_uses_the_full_combat_scope(self) -> None:
        for mode in ("standard", "deep", "wide", None):
            with self.subTest(search_mode=mode):
                engine = floor_reach_eval_module._build_engine(  # noqa: SLF001
                    _Client(),
                    state=floor_reach_eval_module._RunState(),  # noqa: SLF001
                    seed=1,
                    use_beam=True,
                    search_mode=mode,
                    beam_max_depth=None,
                )
                self.assertEqual(
                    engine.beam_search.config.beam_searchable_action_types,
                    COMBAT_BEAM_ACTION_TYPES,
                )

    def test_floor_reach_eval_beam_depth_override_keeps_the_full_combat_scope(self) -> None:
        engine = floor_reach_eval_module._build_engine(  # noqa: SLF001
            _Client(),
            state=floor_reach_eval_module._RunState(),  # noqa: SLF001
            seed=1,
            use_beam=True,
            search_mode="standard",
            beam_max_depth=4,
        )

        config = engine.beam_search.config
        self.assertEqual(config.beam_searchable_action_types, COMBAT_BEAM_ACTION_TYPES)
        self.assertEqual(config.max_depth, 4)

    def test_floor_reach_eval_preserves_explicit_scope(self) -> None:
        explicit = BeamSearchConfig(
            max_depth=2,
            beam_searchable_action_types=frozenset({"system", "card"}),
        )
        engine = floor_reach_eval_module._build_engine(  # noqa: SLF001
            _Client(),
            state=floor_reach_eval_module._RunState(),  # noqa: SLF001
            seed=1,
            use_beam=True,
            search_mode=explicit,
            beam_max_depth=None,
        )

        self.assertEqual(
            engine.beam_search.config.beam_searchable_action_types,
            frozenset({"system", "card"}),
        )

    def test_stable_pruner_ab_cli_config_uses_the_full_combat_scope(self) -> None:
        args = argparse.Namespace(search_mode="standard", beam_depth=None)

        config = stable_pruner_ab_module._cli_beam_config(args)  # noqa: SLF001

        self.assertEqual(config.beam_searchable_action_types, COMBAT_BEAM_ACTION_TYPES)


class SelfPlayBeamScopeTest(unittest.IsolatedAsyncioTestCase):
    async def test_self_play_engine_uses_the_full_combat_scope(self) -> None:
        captured: dict[str, BeamSearchConfig] = {}

        class _RecordingEngine:
            def __init__(self, client, *, beam_config, **kwargs):
                del client, kwargs
                captured["beam_config"] = beam_config

        class _Connection:
            async def connect(self) -> None:
                return None

        class _ApiClient:
            def __init__(self, connection, *, selection_logger=None):
                del connection, selection_logger

            async def close(self) -> None:
                return None

        async def _fake_start_new_run(*args, **kwargs):
            del args, kwargs
            return object()

        with TemporaryDirectory() as tmp:
            with (
                mock.patch.object(self_play_module, "CombatDecisionEngine", _RecordingEngine),
                mock.patch.object(self_play_module, "AsyncTrainingApiClient", _ApiClient),
                mock.patch.object(self_play_module, "start_new_run", _fake_start_new_run),
                mock.patch.object(self_play_module, "_run_result_event", lambda **kw: {}),
            ):
                await self_play_module._run_one(  # noqa: SLF001
                    "run-0",
                    seed=7,
                    connection_factory=_Connection,
                    character_id="IRONCLAD",
                    ascension=0,
                    decision_timeout_s=5.0,
                    max_decisions=10,
                    search_mode="standard",
                    beam_max_depth=None,
                    output_dir=Path(tmp),
                    god_mode=False,
                )

        self.assertIn("beam_config", captured)
        self.assertEqual(
            captured["beam_config"].beam_searchable_action_types,
            COMBAT_BEAM_ACTION_TYPES,
        )


if __name__ == "__main__":
    unittest.main()
