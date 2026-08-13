"""Public Combat Value training-data API with trajectory-chain validation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from sts2_training.decision import _value_training_data_impl as _impl
from sts2_training.decision._value_training_data_impl import *  # noqa: F403


def _validate_episode_chain(episode: CombatValueRLEpisode) -> None:  # noqa: F405
    steps = episode.steps
    for previous, following in zip(steps, steps[1:]):
        if previous.next_decision_point_id != following.decision_point_id:
            raise ValueError(
                "actual trajectory has broken decision_point_id chain "
                f"between decision_index {previous.decision_index} and {following.decision_index}"
            )
        if dict(previous.next_masked_emulator_dto) != dict(following.masked_emulator_dto):
            raise ValueError(
                "actual trajectory has broken masked_emulator_dto chain "
                f"between decision_index {previous.decision_index} and {following.decision_index}"
            )


def load_combat_value_rl_episodes(
    paths: Iterable[str | Path],
    *,
    completed_only: bool = False,
) -> list[CombatValueRLEpisode]:  # noqa: F405
    """Load actual trajectories and reject any broken adjacent transition chain."""

    episodes = _impl.load_combat_value_rl_episodes(paths, completed_only=False)
    for episode in episodes:
        _validate_episode_chain(episode)
    if completed_only:
        return [episode for episode in episodes if episode.usable_for_terminal_return]
    return episodes


__all__ = list(_impl.__all__)
