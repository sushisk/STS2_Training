from __future__ import annotations

import unittest
from types import SimpleNamespace

from sts2_training.runner.stable_pruner_ab_stats import paired_outcome_statistics


class PairedOutcomeStatisticsTest(unittest.TestCase):
    def test_excludes_ties_and_unknown_outcomes_from_discordant_test(self) -> None:
        pairs = tuple(
            SimpleNamespace(winner=winner)
            for winner in ("learned", "learned", "baseline", "tie", None)
        )

        stats = paired_outcome_statistics(pairs)

        self.assertEqual(stats.discordant_pairs, 3)
        self.assertEqual(stats.learned_wins, 2)
        self.assertEqual(stats.baseline_wins, 1)
        self.assertAlmostEqual(stats.learned_share_of_discordant, 2 / 3)
        self.assertEqual(stats.two_sided_exact_sign_test_p_value, 1.0)

    def test_all_discordant_pairs_favoring_one_arm_has_exact_two_sided_p_value(self) -> None:
        pairs = tuple(SimpleNamespace(winner="learned") for _ in range(6))

        stats = paired_outcome_statistics(pairs)

        self.assertEqual(stats.discordant_pairs, 6)
        self.assertEqual(stats.learned_share_of_discordant, 1.0)
        self.assertEqual(stats.two_sided_exact_sign_test_p_value, 0.03125)

    def test_no_discordant_pairs_has_no_test_statistic(self) -> None:
        pairs = (SimpleNamespace(winner="tie"), SimpleNamespace(winner=None))

        stats = paired_outcome_statistics(pairs)

        self.assertEqual(stats.discordant_pairs, 0)
        self.assertIsNone(stats.learned_share_of_discordant)
        self.assertIsNone(stats.two_sided_exact_sign_test_p_value)


if __name__ == "__main__":
    unittest.main()
