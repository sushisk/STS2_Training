"""Small dependency-free statistics for paired stable-pruner A/B outcomes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import Protocol


class _PairWithWinner(Protocol):
    winner: str | None


@dataclass(frozen=True)
class PairedOutcomeStatistics:
    """Exact sign-test summary over discordant win/loss pairs.

    Ties and unknown outcomes are intentionally excluded from the null test. The test asks
    whether learned-vs-baseline wins are symmetric among pairs whose terminal outcomes
    differ; it does not define a promotion threshold and it says nothing about search cost.
    """

    discordant_pairs: int
    learned_wins: int
    baseline_wins: int
    learned_share_of_discordant: float | None
    two_sided_exact_sign_test_p_value: float | None


def paired_outcome_statistics(
    pairs: Sequence[_PairWithWinner],
) -> PairedOutcomeStatistics:
    learned_wins = sum(pair.winner == "learned" for pair in pairs)
    baseline_wins = sum(pair.winner == "baseline" for pair in pairs)
    discordant = learned_wins + baseline_wins
    if discordant == 0:
        return PairedOutcomeStatistics(
            discordant_pairs=0,
            learned_wins=0,
            baseline_wins=0,
            learned_share_of_discordant=None,
            two_sided_exact_sign_test_p_value=None,
        )

    smaller = min(learned_wins, baseline_wins)
    lower_tail_count = sum(comb(discordant, index) for index in range(smaller + 1))
    two_sided = min(Fraction(1, 1), Fraction(2 * lower_tail_count, 2**discordant))
    return PairedOutcomeStatistics(
        discordant_pairs=discordant,
        learned_wins=learned_wins,
        baseline_wins=baseline_wins,
        learned_share_of_discordant=learned_wins / discordant,
        two_sided_exact_sign_test_p_value=float(two_sided),
    )
