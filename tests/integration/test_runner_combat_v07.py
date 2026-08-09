"""Real-RL coverage for `sts2_training.runner.start_combat_from_state`: unlike
`tests/runner/`'s mock-based tests, this verifies `CombatScenario.to_instance_config()`
actually matches what STS2_RL's `build_scenario_from_spec()` expects on the wire, and
that a full episode (start_instance -> decide/commit loop -> close_instance) completes
against the real paired RL server (see `_paired_rl_helpers.py`).
"""

from __future__ import annotations

import asyncio

import pytest

from sts2_training.api import AsyncTrainingApiClient, TcpConnection
from sts2_training.runner import CombatScenario, EnemyScenario, start_combat_from_state

from _paired_rl_helpers import HOST, start_paired_rl, stop_paired_rl

pytestmark = pytest.mark.integration

_LARGE_RESPONSE_LIMIT = 64 * 1024 * 1024


def _one_hit_kill_scenario() -> CombatScenario:
    # WHIRLWIND hits every enemy with no target choice needed, so a single decision
    # (whichever policy/fallback picks the only card in hand) reliably ends combat -
    # this test is about the runner's plumbing, not about exercising real strategy.
    return CombatScenario(
        character_id="IRONCLAD",
        player_hp=80,
        player_max_hp=80,
        hand=["WHIRLWIND"],
        draw_pile=[],
        discard_pile=[],
        enemies=[EnemyScenario(monster_id="CALCIFIED_CULTIST", hp=1)],
    )


async def _run(port: int) -> None:
    connection = TcpConnection(host=HOST, port=port, connect_timeout_s=5.0, max_response_bytes=_LARGE_RESPONSE_LIMIT)
    async with AsyncTrainingApiClient(connection) as client:
        result = await start_combat_from_state(
            client,
            _one_hit_kill_scenario(),
            decision_timeout_s=30.0,
            max_decisions=5,
        )

    assert result.decisions_made >= 1
    assert result.final_dto["legal_actions"] == []
    assert result.final_dto["outcome"] == "victory"


def test_start_combat_from_state_completes_a_real_episode() -> None:
    process, port = start_paired_rl(pytest.skip)
    try:
        asyncio.run(_run(port))
    finally:
        stop_paired_rl(process)
