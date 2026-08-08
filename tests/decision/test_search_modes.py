from __future__ import annotations

import unittest

from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.decision.search_modes import (
    DEFAULT_SEARCH_MODE,
    SEARCH_MODES,
    resolve_search_mode,
)


class ResolveSearchModeTest(unittest.TestCase):
    def test_none_resolves_to_default_mode(self) -> None:
        self.assertIs(resolve_search_mode(None), SEARCH_MODES[DEFAULT_SEARCH_MODE])

    def test_named_mode_resolves_to_its_preset(self) -> None:
        self.assertIs(resolve_search_mode("deep"), SEARCH_MODES["deep"])

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

    def test_every_mode_is_a_valid_beam_search_config(self) -> None:
        for name, config in SEARCH_MODES.items():
            with self.subTest(mode=name):
                self.assertIsInstance(config, BeamSearchConfig)
                self.assertGreater(config.max_depth, 0)
                self.assertGreater(config.beam_width, 0)
                self.assertGreater(config.top_k_actions, 0)


if __name__ == "__main__":
    unittest.main()
