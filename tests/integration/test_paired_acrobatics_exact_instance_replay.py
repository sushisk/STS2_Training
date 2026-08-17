from __future__ import annotations

import asyncio

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

# These fields are intentionally outside the Training-visible decision contract. This
# regression guards that boundary only; it must not use their presence/absence to infer
# how RL proves or materializes replay correctness internally.
_FORBIDDEN_REPLAY_PROVENANCE_KEYS = frozenset(
    {"cardInstanceId", "card_instance_id", "cardsDrawnThisStep"}
)


def _acrobatics_config() -> dict:
    """Reproduce the known mixed-hand Acrobatics path using ordinary public state."""

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


def _assert_no_replay_provenance_metadata(
    value: object, *, path: str = "masked_emulator_dto"
) -> None:
    """Fail if replay-provenance-only metadata crosses the Training boundary."""

    if isinstance(value, dict):
        leaked_keys = _FORBIDDEN_REPLAY_PROVENANCE_KEYS.intersection(value)
        assert not leaked_keys, (
            f"Training-visible replay provenance leaked at {path}: "
            f"{sorted(leaked_keys)!r}"
        )
        for key, child in value.items():
            _assert_no_replay_provenance_metadata(child, path=f"{path}.{key}")
        return

    if isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_replay_provenance_metadata(child, path=f"{path}[{index}]")


def _visible_choice_card_ids(decision: dict) -> list[str]:
    """Read only normal Training-visible choice semantics."""

    masked = decision["masked_emulator_dto"]
    _assert_no_replay_provenance_metadata(masked)

    pending_choice = masked.get("pendingChoice") or {}
    options = pending_choice.get("options") or []
    assert len(options) == 4, options

    card_ids: list[str] = []
    choice_actions = [
        action
        for action in _available_actions(decision)
        if action.get("action_type") == "choice_card"
    ]
    assert len(choice_actions) == 4, choice_actions
    for action in choice_actions:
        parameters = action.get("parameters") or {}
        card_id = parameters.get("cardId")
        assert isinstance(card_id, str) and card_id
        card_ids.append(card_id)

    return card_ids


async def _exercise_acrobatics_training_boundary(port: int) -> None:
    connection = TcpConnection(host=_HOST, port=port, connect_timeout_s=5.0)
    async with AsyncTrainingApiClient(connection) as client:
        instance_id = await client.start_instance(_acrobatics_config(), timeout_s=30.0)
        root = await client.get_decision(instance_id, timeout_s=30.0)
        acrobatics = _find_card_action(root, "ACROBATICS")

        # Commit through the ordinary Training -> RL -> Emulator path. Training checks
        # only the public decision it receives; whether RL used any special replay path is
        # deliberately an implementation detail outside this test.
        pending = await client.commit_action(
            instance_id,
            root["decision_point_id"],
            acrobatics["action_id"],
            timeout_s=30.0,
        )
        card_ids = _visible_choice_card_ids(pending)
        assert len(card_ids) == 4, card_ids
        assert set(card_ids) >= {
            "NEUTRALIZE",
            "DEFEND_SILENT",
            "STRIKE_SILENT",
        }
        assert card_ids.count("DEFEND_SILENT") == 2, card_ids

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

        # The Training acceptance boundary is observable search behavior: an admitted /
        # emulated replay failure must surface as a branch fault rather than disappear.
        # For this known-good paired case, therefore, no such fault should be present.
        assert collected.search_result.stats.branches_faulted == 0
        replay_mismatches = [
            event
            for event in collected.trace
            if isinstance(event, BranchFaultTrace)
            and event.fault_kind == "replay_mismatch"
        ]
        assert replay_mismatches == []

        # Multiple hypotheses make the regression exercise more than one search branch.
        # Their rng_id values are not evidence for replay correctness and are not used to
        # infer any provenance/transfer/order property.
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


def test_paired_acrobatics_has_no_branch_faults() -> None:
    """Cross-repo regression for Training's fault-visibility and label-safety boundary.

    The test is skipped unless STS2_RL_ROOT points at the paired RL checkout. It does not
    reproduce RL's DrawPile transfer/order proof or inspect any replay materialization
    details. It verifies that the ordinary Training-visible choice remains usable and that
    the production Oracle/Beam path completes without replay_mismatch branch faults.
    """

    process, port = _start_paired_rl(pytest.skip)
    try:
        asyncio.run(_exercise_acrobatics_training_boundary(port))
    finally:
        _stop_paired_rl(process)
