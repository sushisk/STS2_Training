from __future__ import annotations

import asyncio

import pytest

from sts2_training.api import AsyncTrainingApiClient, TcpConnection, TransportError

from _paired_rl_helpers import HOST as _HOST
from _paired_rl_helpers import start_paired_rl as _start_paired_rl
from _paired_rl_helpers import stop_paired_rl as _stop_paired_rl

pytestmark = pytest.mark.integration

_SERVER_MAX_MESSAGE_BYTES = 4096
_PAIRED_MAX_EMULATE_ACTIONS_ITEMS = 16
_LARGE_RESPONSE_LIMIT = 64 * 1024 * 1024


def _combat_config() -> dict:
    return {
        "instance_type": "combat",
        "character_id": "IRONCLAD",
        "player_hp": None,
        "player_max_hp": None,
        "hand": ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"],
        "draw_pile": [],
        "discard_pile": [],
        "exhaust_pile": [],
        "player_powers": [],
        "relics": [],
        "potions": [],
        "seed": 1,
        "enemies": [{"monster_id": "CALCIFIED_CULTIST", "hp": 48}],
    }


def _action_id(decision: dict) -> str:
    actions = decision["masked_emulator_dto"]["legal_actions"]
    for action in actions:
        params = action.get("parameters") or {}
        if params.get("cardId") == "DEFEND_IRONCLAD":
            return action["action_id"]
    return actions[0]["action_id"]


async def _assert_request_size_boundary(port: int) -> None:
    session_id = "paired-size-boundary"
    connection = TcpConnection(
        host=_HOST,
        port=port,
        client_session_id=session_id,
        connect_timeout_s=5.0,
        max_message_bytes=_SERVER_MAX_MESSAGE_BYTES * 4,
        max_response_bytes=_LARGE_RESPONSE_LIMIT,
    )
    oversized = {
        "schema_version": "0.7",
        "client_session_id": session_id,
        "request_seq": 1,
        "request_id": f"{session_id}:1",
        "operation": "get_decision",
        "instance_id": "not-reached",
        "branch_id": "root",
        "padding": "x" * (_SERVER_MAX_MESSAGE_BYTES * 2),
    }
    async with connection:
        with pytest.raises(TransportError, match="message_too_large") as exc_info:
            await connection.exchange(oversized, timeout_s=5.0)
        assert exc_info.value.completion_uncertain is False


async def _exercise_paired_v07(port: int) -> None:
    await _assert_request_size_boundary(port)

    connection = TcpConnection(
        host=_HOST,
        port=port,
        connect_timeout_s=5.0,
        max_response_bytes=_LARGE_RESPONSE_LIMIT,
    )
    async with AsyncTrainingApiClient(connection) as client:
        instance_id = await client.start_instance(_combat_config(), timeout_s=30.0)
        assert client.max_emulate_actions_items == _PAIRED_MAX_EMULATE_ACTIONS_ITEMS
        root = await client.get_decision(instance_id, timeout_s=30.0)
        root_action = _action_id(root)

        # The discovered RL capability is enforced locally before a wire request can
        # consume request_seq. The paired server intentionally uses non-default 16.
        seq_before_capacity = client.next_request_seq
        with pytest.raises(ValueError, match="max_emulate_actions_items=16"):
            await client.emulate_actions(
                instance_id,
                [
                    {
                        "parent_branch_id": "root",
                        "branch_id": f"too-wide-{index}",
                        "rng_id": index + 1,
                        "decision_point_id": root["decision_point_id"],
                        "action_id": root_action,
                    }
                    for index in range(_PAIRED_MAX_EMULATE_ACTIONS_ITEMS + 1)
                ],
                timeout_s=30.0,
            )
        assert client.next_request_seq == seq_before_capacity

        # The Training contract can reject same-batch parent dependencies completely
        # locally, in either repository pairing, without consuming request_seq.
        seq_before_invalid = client.next_request_seq
        with pytest.raises(ValueError, match="created within the same batch"):
            await client.emulate_actions(
                instance_id,
                [
                    {
                        "parent_branch_id": "root",
                        "branch_id": "same-parent",
                        "rng_id": 1,
                        "decision_point_id": root["decision_point_id"],
                        "action_id": root_action,
                    },
                    {
                        "parent_branch_id": "same-parent",
                        "branch_id": "same-child",
                        "rng_id": 1,
                        "decision_point_id": root["decision_point_id"],
                        "action_id": root_action,
                    },
                ],
                timeout_s=30.0,
            )
        assert client.next_request_seq == seq_before_invalid

        prepared = await client.emulate_actions(
            instance_id,
            [
                {
                    "parent_branch_id": "root",
                    "branch_id": "b1",
                    "rng_id": 1,
                    "decision_point_id": root["decision_point_id"],
                    "action_id": root_action,
                },
                {
                    "parent_branch_id": "root",
                    "branch_id": "b2",
                    "rng_id": 2,
                    "decision_point_id": root["decision_point_id"],
                    "action_id": root_action,
                },
            ],
            timeout_s=30.0,
        )
        b1 = prepared["branch_results"]["b1"]
        b2 = prepared["branch_results"]["b2"]
        assert b1["status"] == "completed"
        assert b2["status"] == "completed"

        target_items = [
            {
                "parent_branch_id": "b1",
                "branch_id": "c1",
                "rng_id": 1,
                "decision_point_id": b1["decision_point_id"],
                "action_id": _action_id(b1),
            },
            {
                "parent_branch_id": "b2",
                "branch_id": "forced-fault",
                "rng_id": 2,
                "decision_point_id": b2["decision_point_id"],
                "action_id": _action_id(b2),
            },
        ]

        # Force a response-frame size failure after RL has executed and cached the batch.
        # The exact pending request must then replay successfully after increasing only
        # the local receive bound. Re-execution would collide with burned branch IDs.
        await connection.set_max_response_bytes(32)
        seq_before_uncertain = client.next_request_seq
        with pytest.raises(TransportError):
            await client.emulate_actions(
                instance_id,
                target_items,
                timeout_s=30.0,
            )
        pending = client.pending_retry
        assert pending is not None
        assert pending.operation == "emulate_actions"
        assert pending.request_seq == seq_before_uncertain
        assert client.next_request_seq == seq_before_uncertain

        await connection.set_max_response_bytes(_LARGE_RESPONSE_LIMIT)
        replayed = await client.retry_request(pending, timeout_s=30.0)
        assert replayed["status"] == "completed"
        assert replayed["branch_results"]["c1"]["status"] == "completed"
        forced = replayed["branch_results"]["forced-fault"]
        assert forced["status"] == "faulted"
        assert forced["fault_kind"] == "integration_test"
        assert client.pending_retry is None
        assert client.next_request_seq == seq_before_uncertain + 1

        await client.close_instance(instance_id, timeout_s=30.0)


def test_paired_rl_v07_multi_parent_mixed_fault_and_exact_replay() -> None:
    process, port = _start_paired_rl(pytest.skip)
    try:
        asyncio.run(_exercise_paired_v07(port))
    finally:
        _stop_paired_rl(process)
