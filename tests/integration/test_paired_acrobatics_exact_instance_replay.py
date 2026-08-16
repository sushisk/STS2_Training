from __future__ import annotations

import asyncio
import re

import pytest

from sts2_training.api import AsyncTrainingApiClient, TcpConnection
from sts2_training.decision.beam_search import BeamSearchConfig
from sts2_training.decision.combat_decision import COMBAT_BEAM_ACTION_TYPES
from sts2_training.decision.oracle_search import OracleCollectionConfig
from sts2_training.decision.oracle_value_logging import RootValueLoggingOracleCollector
from sts2_training.decision.policy import PriorHeuristicPolicy
from sts2_training.decision.search_trace import BranchFaultTrace, PolicyProposalTrace
from sts2_training.decision.value import HeuristicValueFunction

from _paired_rl_helpers import HOST as _HOST
from _paired_rl_helpers import start_paired_rl as _start_paired_rl
from _paired_rl_helpers import stop_paired_rl as _stop_paired_rl

pytestmark = pytest.mark.integration

_CARD_INSTANCE_ID = re.compile(r"cardv-[0-9a-f]{32}\Z")


def _acrobatics_config() -> dict:
    """Mirror Emulator #25's duplicate-CardId case with a normal mixed hand.

    The pre-existing NEUTRALIZE is deliberate: Acrobatics asks the player to discard
    from the whole post-draw hand, not only from the three cards it just drew. A paired
    replay fix must therefore remain correct when the visible PendingChoice contains a
    card that was already in Hand before the real Acrobatics step.
    """

    return {
        "instance_type": "combat",
        "character_id": "SILENT",
        "player_hp": 70,
        "player_max_hp": 70,
        "hand_cards": [
            {"card_id": "ACROBATICS", "is_upgraded": False},
            {"card_id": "NEUTRALIZE", "is_upgraded": False},
        ],
        "draw_pile_cards": [
            {"card_id": "DEFEND_SILENT", "is_upgraded": True},
            {"card_id": "DEFEND_SILENT", "is_upgraded": False},
            {"card_id": "STRIKE_SILENT", "is_upgraded": False},
            {"card_id": "SURVIVOR", "is_upgraded": False},
        ],
        "discard_pile": [],
        "exhaust_pile": [],
        "player_powers": [],
        "relics": [],
        "potions": [],
        "seed": 1,
        "enemies": [
            {
                "monster_id": "CALCIFIED_CULTIST",
                "hp": 999,
                "max_hp": 999,
            }
        ],
    }


def _available_actions(decision: dict) -> list[dict]:
    return [
        action
        for action in decision["masked_emulator_dto"]["legal_actions"]
        if action.get("is_available") is not False
    ]


def _find_card_action(decision: dict, card_id: str) -> dict:
    for action in _available_actions(decision):
        if action.get("action_type") != "card":
            continue
        if (action.get("parameters") or {}).get("cardId") == card_id:
            return action
    raise AssertionError(
        f"missing playable card {card_id!r}: {_available_actions(decision)!r}"
    )


def _visible_choice_identities(decision: dict) -> list[tuple[str, str]]:
    identities: list[tuple[str, str]] = []
    for action in _available_actions(decision):
        if action.get("action_type") != "choice_card":
            continue
        parameters = action.get("parameters") or {}
        card_id = parameters.get("cardId")
        card_instance_id = parameters.get("cardInstanceId")
        assert isinstance(card_id, str) and card_id
        assert (
            isinstance(card_instance_id, str)
            and _CARD_INSTANCE_ID.fullmatch(card_instance_id)
        )
        identities.append((card_id, card_instance_id))
    return identities


async def _exercise_acrobatics_exact_instance_replay(port: int) -> None:
    connection = TcpConnection(host=_HOST, port=port, connect_timeout_s=5.0)
    async with AsyncTrainingApiClient(connection) as client:
        instance_id = await client.start_instance(_acrobatics_config(), timeout_s=30.0)
        root = await client.get_decision(instance_id, timeout_s=30.0)
        acrobatics = _find_card_action(root, "ACROBATICS")

        # This is a real committed step, not an emulated branch. Emulator #25 first
        # publishes the concrete card identities at the resulting visible choice, and
        # RL #64 is responsible for translating that evidence into replay constraints.
        pending = await client.commit_action(
            instance_id,
            root["decision_point_id"],
            acrobatics["action_id"],
            timeout_s=30.0,
        )
        identities = _visible_choice_identities(pending)
        assert len(identities) == 4, identities
        assert len({public_id for _card_id, public_id in identities}) == 4, identities
        assert {card_id for card_id, _public_id in identities} >= {
            "NEUTRALIZE",
            "DEFEND_SILENT",
            "STRIKE_SILENT",
        }
        defend_instance_ids = [
            public_id
            for card_id, public_id in identities
            if card_id == "DEFEND_SILENT"
        ]
        assert len(defend_instance_ids) == 2, identities
        assert defend_instance_ids[0] != defend_instance_ids[1], identities

        # Training intentionally does not derive, decode, or compare these opaque
        # cardv-* values with Snapshot InstanceId. The only identity assertion here is
        # public-domain shape/distinctness at the already-visible choice boundary.
        collector = RootValueLoggingOracleCollector(
            client,
            policy=PriorHeuristicPolicy(),
            value_fn=HeuristicValueFunction(),
            config=OracleCollectionConfig(
                beam_config=BeamSearchConfig(
                    beam_width=8,
                    top_k_actions=8,
                    max_depth=1,
                    max_batch_size=16,
                    beam_searchable_action_types=COMBAT_BEAM_ACTION_TYPES,
                ),
                target_beam_width=4,
                exhaustive_root_actions=True,
            ),
        )
        collected = await collector.collect(instance_id, pending, timeout_s=60.0)

        assert collected.search_result.stats.branches_faulted == 0
        replay_mismatches = [
            event
            for event in collected.trace
            if isinstance(event, BranchFaultTrace)
            and event.fault_kind == "replay_mismatch"
        ]
        assert replay_mismatches == []

        root_proposals = [
            event
            for event in collected.trace
            if isinstance(event, PolicyProposalTrace)
            and event.parent_branch_id == "root"
        ]
        assert len(root_proposals) == 1
        rng_ids = {candidate.rng_id for candidate in root_proposals[0].candidates}
        assert len(rng_ids) >= 2, root_proposals[0].candidates

        await client.close_instance(instance_id, timeout_s=30.0)


def test_paired_acrobatics_exact_instance_replay_has_no_branch_faults() -> None:
    """Cross-repo gate for Training #72 + RL #64 + Emulator #25.

    The test is skipped unless STS2_RL_ROOT points at the paired RL checkout. That RL
    checkout must in turn use an Emulator build containing #25 (or the equivalent merged
    implementation). It exercises the production Training Oracle/Beam path rather than
    calling RL's CombatInstance directly.
    """

    process, port = _start_paired_rl(pytest.skip)
    try:
        asyncio.run(_exercise_acrobatics_exact_instance_replay(port))
    finally:
        _stop_paired_rl(process)
