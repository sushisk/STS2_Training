from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sts2_training.api import AsyncTrainingApiClient, TcpConnection, TransportError

pytestmark = pytest.mark.integration

_HOST = "127.0.0.1"
_LARGE_RESPONSE_LIMIT = 64 * 1024 * 1024
_SERVER_HELPER = Path(__file__).with_name("_paired_rl_server_v07.py")


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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_HOST, 0))
        return int(sock.getsockname()[1])


def _start_paired_rl() -> tuple[subprocess.Popen[str], int]:
    root_value = os.environ.get("STS2_RL_ROOT")
    if not root_value:
        pytest.skip("set STS2_RL_ROOT to the paired STS2_RL checkout")
    root = Path(root_value).resolve()
    if not (root / "API" / "tcp_server.py").is_file():
        pytest.skip(f"STS2_RL_ROOT is not a usable RL checkout: {root}")

    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, str(_SERVER_HELPER), str(root), _HOST, str(port)],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout is not None else ""
            raise AssertionError(f"paired RL exited before startup:\n{output}")
        try:
            with socket.create_connection((_HOST, port), timeout=0.2):
                return process, port
        except OSError:
            time.sleep(0.1)

    process.terminate()
    output = process.communicate(timeout=5)[0]
    raise AssertionError(f"paired RL did not start within 30s:\n{output}")


def _stop_paired_rl(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


async def _exercise_paired_v07(port: int) -> None:
    connection = TcpConnection(
        host=_HOST,
        port=port,
        connect_timeout_s=5.0,
        max_response_bytes=_LARGE_RESPONSE_LIMIT,
    )
    async with AsyncTrainingApiClient(connection) as client:
        instance_id = await client.start_instance(_combat_config(), timeout_s=30.0)
        root = await client.get_decision(instance_id, timeout_s=30.0)
        root_action = _action_id(root)

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
    process, port = _start_paired_rl()
    try:
        asyncio.run(_exercise_paired_v07(port))
    finally:
        _stop_paired_rl(process)
