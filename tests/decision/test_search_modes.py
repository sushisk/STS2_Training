from __future__ import annotations

import unittest

from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.decision.search_modes import (
    DEFAULT_SEARCH_MODE,
    SEARCH_MODES,
    resolve_search_mode,
)


class ResolveSearchModeTest(unittest.TestCase):
    def test_none_resolves_to_isolated_default_mode(self) -> None:
        resolved = resolve_search_mode(None)

        self.assertIsNot(resolved, SEARCH_MODES[DEFAULT_SEARCH_MODE])
        self.assertEqual(resolved, SEARCH_MODES[DEFAULT_SEARCH_MODE])

    def test_named_mode_resolves_to_isolated_preset(self) -> None:
        resolved = resolve_search_mode("deep")

        self.assertIsNot(resolved, SEARCH_MODES["deep"])
        self.assertEqual(resolved, SEARCH_MODES["deep"])
        self.assertIsNot(resolved.simulation_options, SEARCH_MODES["deep"].simulation_options)

    def test_mutating_resolved_named_mode_does_not_corrupt_preset(self) -> None:
        resolved = resolve_search_mode("deep")
        original_depth = SEARCH_MODES["deep"].max_depth
        original_options = dict(SEARCH_MODES["deep"].simulation_options or {})

        resolved.max_depth = 99
        assert resolved.simulation_options is not None
        resolved.simulation_options["stop_condition"] = "combat_end"

        self.assertEqual(SEARCH_MODES["deep"].max_depth, original_depth)
        self.assertEqual(SEARCH_MODES["deep"].simulation_options, original_options)

    def test_unknown_mode_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_search_mode("nonexistent")

    def test_explicit_config_is_used_as_is(self) -> None:
        custom = BeamSearchConfig(max_depth=9, beam_width=1, top_k_actions=1)

        self.assertIs(resolve_search_mode(custom), custom)

    def test_max_depth_overrides_named_mode_without_mutating_preset(self) -> None:
        original_depth = SEARCH_MODES["shallow"].max_depth

        resolved = resolve_search_mode("shallow", max_depth=7)

        self.assertEqual(resolved.max_depth, 7)
        self.assertEqual(resolved.beam_width, SEARCH_MODES["shallow"].beam_width)
        self.assertEqual(SEARCH_MODES["shallow"].max_depth, original_depth)

    def test_max_depth_overrides_explicit_config(self) -> None:
        custom = BeamSearchConfig(max_depth=1, beam_width=2, top_k_actions=2)

        resolved = resolve_search_mode(custom, max_depth=5)

        self.assertEqual(resolved.max_depth, 5)
        self.assertEqual(resolved.beam_width, 2)
        self.assertIsNot(resolved.simulation_options, custom.simulation_options)

    def test_standard_mode_tracks_beam_search_defaults(self) -> None:
        self.assertEqual(SEARCH_MODES["standard"], BeamSearchConfig())

    def test_every_mode_is_a_valid_beam_search_config(self) -> None:
        for name, config in SEARCH_MODES.items():
            with self.subTest(mode=name):
                self.assertIsInstance(config, BeamSearchConfig)
                self.assertGreater(config.max_depth, 0)
                self.assertGreater(config.beam_width, 0)
                self.assertGreater(config.top_k_actions, 0)


if __name__ == "__main__":
    unittest.main()
