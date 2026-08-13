"""Real-RL coverage for `sts2_training.runner.start_combat_from_state` under the
paired Training↔RL wire DTO v0.8 contract: unlike `tests/runner/`'s mock-based tests,
this verifies `CombatScenario.to_instance_config()` actually matches what STS2_RL's
`build_scenario_from_spec()` expects on the wire, and that a full episode
(start_instance -> decide/commit loop -> close_instance) completes against the real
paired RL server (see `_paired_rl_helpers.py`).
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


def _upgrade_and_enchantment_survival_scenario() -> CombatScenario:
    # No card is ever played here - this test only checks that per-instance
    # upgrade_level/enchantment data set on a scenario's hand round-trips through
    # the real v0.8 wire (emulator -> RL TCP server -> Training client) unmangled,
    # via a single get_decision() with no commit.
    return CombatScenario(
        character_id="IRONCLAD",
        player_hp=80,
        player_max_hp=80,
        hand=[],
        draw_pile=[],
        discard_pile=[],
        enemies=[EnemyScenario(monster_id="CALCIFIED_CULTIST", hp=48)],
        extra={
            "hand_cards": [
                {"card_id": "WITHER", "upgrade_level": 1},
                {"card_id": "STRIKE_IRONCLAD", "enchantment": {"id": "SHARP", "amount": 3}},
            ],
        },
    )


async def _fetch_initial_dto(port: int) -> dict:
    connection = TcpConnection(host=HOST, port=port, connect_timeout_s=5.0, max_response_bytes=_LARGE_RESPONSE_LIMIT)
    async with AsyncTrainingApiClient(connection) as client:
        instance_id = await client.start_instance(
            _upgrade_and_enchantment_survival_scenario().to_instance_config(), timeout_s=30.0
        )
        decision = await client.get_decision(instance_id, timeout_s=30.0)
        await client.close_instance(instance_id, timeout_s=10.0)
    return decision["masked_emulator_dto"]


def test_wither_upgrade_level_and_sharp_enchantment_survive_the_real_wire() -> None:
    process, port = start_paired_rl(pytest.skip)
    try:
        dto = asyncio.run(_fetch_initial_dto(port))
    finally:
        stop_paired_rl(process)

    assert dto["mask_version"] == "1.2"

    hand = dto["hand"]
    wither_cards = [c for c in hand if c.get("id") == "WITHER"]
    assert wither_cards, hand
    assert wither_cards[0]["upgraded"] is True, wither_cards[0]
    assert wither_cards[0]["upgradeLevel"] == 1, wither_cards[0]

    strike_cards = [c for c in hand if c.get("id") == "STRIKE_IRONCLAD"]
    assert strike_cards, hand
    enchantment = strike_cards[0].get("enchantment")
    assert enchantment is not None and enchantment.get("id") == "SHARP" and enchantment.get("amount") == 3, strike_cards[0]
