"""Cross-repository regressions for canonical CardInstances snapshot restoration.

The assertions deliberately use only the Training-visible API.  ``emulate_action``
executes in an RL branch worker, which restores the root's held Emulator snapshot before
performing the action.  That makes these tests cover the real Training -> RL -> Emulator
restore path without exposing snapshot internals in the Training wire DTO.
"""

from __future__ import annotations

import asyncio

import pytest

from sts2_training.api import AsyncTrainingApiClient, TcpConnection

from _paired_rl_helpers import HOST, start_production_rl, stop_paired_rl

pytestmark = pytest.mark.integration

_LARGE_RESPONSE_LIMIT = 64 * 1024 * 1024


def _config(*, energy: int, hand_cards: list[str]) -> dict:
    return {
        "instance_type": "combat",
        "character_id": "NECROBINDER",
        "player_hp": 70,
        "player_max_hp": 70,
        "energy": energy,
        "hand_cards": [{"card_id": card_id} for card_id in hand_cards],
        "draw_pile_cards": [],
        "discard_pile": [],
        "exhaust_pile": [],
        "player_powers": [],
        "relics": [],
        "potions": [],
        "seed": 1,
        "enemies": [{"monster_id": "CALCIFIED_CULTIST", "hp": 999, "max_hp": 999}],
    }


def _action(decision: dict, action_type: str, card_id: str | None = None) -> dict:
    actions = decision["masked_emulator_dto"]["legal_actions"]
    for action in actions:
        if action.get("action_type") != action_type:
            continue
        if card_id is None or (action.get("parameters") or {}).get("cardId") == card_id:
            return action
    raise AssertionError(f"missing {action_type=} {card_id=}: {actions!r}")


def _card_cost(decision: dict, card_id: str) -> int:
    action = _action(decision, "card", card_id)
    cost = (action.get("parameters") or {}).get("cost")
    assert isinstance(cost, int), action
    return cost


def _pile_count(masked_state: dict, pile_name: str, card_id: str) -> int:
    return sum(
        entry.get("count", 0)
        for entry in masked_state.get(pile_name, [])
        if entry.get("id") == card_id
    )


async def _banshees_cry_history_round_trip(port: int) -> None:
    connection = TcpConnection(
        host=HOST,
        port=port,
        connect_timeout_s=5.0,
        max_response_bytes=_LARGE_RESPONSE_LIMIT,
    )
    async with AsyncTrainingApiClient(connection) as client:
        instance_id = await client.start_instance(
            _config(energy=20, hand_cards=["APPARITION", "BANSHEES_CRY"]),
            timeout_s=30.0,
        )
        root = await client.get_decision(instance_id, timeout_s=30.0)

        # APPARITION is Ethereal. Its completed CardPlayFinishedEntry must survive
        # the held snapshot captured after this commit; malformed WasEthereal history
        # previously rejected Snapshot Restore before this branch could execute.
        apparition = _action(root, "card", "APPARITION")
        after_apparition = await client.commit_action(
            instance_id,
            root["decision_point_id"],
            apparition["action_id"],
            timeout_s=30.0,
        )
        assert after_apparition["status"] == "completed"
        assert _card_cost(after_apparition, "BANSHEES_CRY") == 7

        # This branch worker restores the just-captured held snapshot before playing
        # BansheesCry. Its successful execution proves that the Ethereal history entry
        # remains bound to the same removed Apparition instance during restore.
        banshees_cry = _action(after_apparition, "card", "BANSHEES_CRY")
        branch = await client.emulate_action(
            instance_id,
            "root",
            "banshees-cry-history",
            1,
            after_apparition["decision_point_id"],
            banshees_cry["action_id"],
            timeout_s=60.0,
        )
        assert branch["status"] == "completed"
        # 20 - APPARITION(1) - BANSHEES_CRY(7). This also protects the cost modifier
        # BansheesCry received when the Ethereal card was played.
        assert branch["masked_emulator_dto"]["energy"] == 12

        await client.close_instance(instance_id, timeout_s=30.0)


def test_banshees_cry_uses_ethereal_history_after_snapshot_restore() -> None:
    process, port = start_production_rl(pytest.skip)
    try:
        asyncio.run(_banshees_cry_history_round_trip(port))
    finally:
        stop_paired_rl(process)


async def _sculpting_strike_ethereal_round_trip(port: int) -> None:
    connection = TcpConnection(
        host=HOST,
        port=port,
        connect_timeout_s=5.0,
        max_response_bytes=_LARGE_RESPONSE_LIMIT,
    )
    async with AsyncTrainingApiClient(connection) as client:
        instance_id = await client.start_instance(
            _config(energy=3, hand_cards=["SCULPTING_STRIKE", "STRIKE_IRONCLAD"]),
            timeout_s=30.0,
        )
        root = await client.get_decision(instance_id, timeout_s=30.0)

        # With one eligible card in hand, the engine selects STRIKE_IRONCLAD as
        # SculptingStrike's ActionContinuation and adds Ethereal to that exact instance.
        # The following stable decision therefore has a held snapshot containing the
        # local keyword mutation.
        sculpting_strike = _action(root, "card", "SCULPTING_STRIKE")
        after_sculpting = await client.commit_action(
            instance_id,
            root["decision_point_id"],
            sculpting_strike["action_id"],
            timeout_s=30.0,
        )
        assert after_sculpting["status"] == "completed"
        assert _action(after_sculpting, "card", "STRIKE_IRONCLAD")

        # End Turn in a worker, forcing Snapshot Restore before Ethereal processing.
        # The selected Strike must exhaust; restoring it as a fresh card would instead
        # put it into discard, which is visible in Training's masked pile multisets.
        end_turn = _action(after_sculpting, "system")
        branch = await client.emulate_action(
            instance_id,
            "root",
            "sculpting-strike-ethereal",
            1,
            after_sculpting["decision_point_id"],
            end_turn["action_id"],
            timeout_s=60.0,
        )
        assert branch["status"] == "completed"
        masked = branch["masked_emulator_dto"]
        assert _pile_count(masked, "exhaustPile", "STRIKE_IRONCLAD") == 1
        assert _pile_count(masked, "discardPile", "STRIKE_IRONCLAD") == 0

        await client.close_instance(instance_id, timeout_s=30.0)


def test_sculpting_strike_ethereal_instance_survives_snapshot_restore() -> None:
    process, port = start_production_rl(pytest.skip)
    try:
        asyncio.run(_sculpting_strike_ethereal_round_trip(port))
    finally:
        stop_paired_rl(process)
