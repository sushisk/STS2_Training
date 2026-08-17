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

# Replay reconstruction stays RL-internal. These names cover both discarded public
# protocol designs and the current #64 ReplayPrefix bookkeeping. None belongs in a
# Training-visible decision DTO.
_FORBIDDEN_REPLAY_INTERNAL_KEYS = frozenset(
    {
        "cardInstanceId",
        "card_instance_id",
        "cardsDrawnThisStep",
        "visibleDrawConstraints",
        "visible_draw_constraints",
        "visibleDrawTrackingBlocked",
        "visible_draw_tracking_blocked",
        "visibleDrawTrackingError",
        "visible_draw_tracking_error",
        "rootRelativeDrawOffset",
        "root_relative_draw_offset",
        "observableCardKey",
        "observable_card_key",
    }
)


def _acrobatics_config() -> dict:
    """Reproduce the public Hand-transfer shape used by the current RL #64 regression.

    The duplicate DEFEND_SILENT copies intentionally differ in upgraded state. Current
    #64 distinguishes replay-relevant public card state structurally; Training must not
    need a physical-copy identity token to exercise the same scenario.
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


def _assert_no_replay_internal_metadata(
    value: object, *, path: str = "masked_emulator_dto"
) -> None:
    """Fail if replay-only evidence/bookkeeping crosses the Training boundary."""

    if isinstance(value, dict):
        leaked_keys = _FORBIDDEN_REPLAY_INTERNAL_KEYS.intersection(value)
        assert not leaked_keys, (
            f"Training-visible replay internals leaked at {path}: "
            f"{sorted(leaked_keys)!r}"
        )
        for key, child in value.items():
            _assert_no_replay_internal_metadata(child, path=f"{path}.{key}")
        return

    if isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_replay_internal_metadata(child, path=f"{path}[{index}]")


def _training_visible_choice_card_ids(decision: dict) -> list[str]:
    """Read only ordinary Training-visible legal-action semantics.

    Current RL #64 derives its replay evidence from committed pre/post public Hand and
    DrawPile state. PendingChoice option ordering/metadata is deliberately not part of
    that proof, so this regression does not inspect or assert it.
    """

    masked = decision["masked_emulator_dto"]
    _assert_no_replay_internal_metadata(masked)

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
        _assert_no_replay_internal_metadata(root["masked_emulator_dto"])
        acrobatics = _find_card_action(root, "ACROBATICS")

        # Commit through the ordinary Training -> RL -> Emulator path. RL #64 may derive
        # Hand-transfer evidence from public pre/post state and record ReplayPrefix
        # constraints internally; Training receives none of that representation.
        pending = await client.commit_action(
            instance_id,
            root["decision_point_id"],
            acrobatics["action_id"],
            timeout_s=30.0,
        )
        card_ids = _training_visible_choice_card_ids(pending)
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

        # Training's acceptance boundary is observable search behavior: an admitted /
        # emulated replay failure must surface as a branch fault rather than disappear.
        # For this known-good paired path, therefore, no such fault should be present.
        assert collected.search_result.stats.branches_faulted == 0
        replay_mismatches = [
            event
            for event in collected.trace
            if isinstance(event, BranchFaultTrace)
            and event.fault_kind == "replay_mismatch"
        ]
        assert replay_mismatches == []

        # Multiple hypotheses make the regression exercise more than one search branch.
        # rng_id is only an exercise dimension here; current #64 explicitly does not use
        # it as draw provenance or replay-evidence input.
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

    The historical filename predates #64's current public structural design. The test is
    skipped unless STS2_RL_ROOT points at the paired RL checkout. It intentionally does
    not reproduce RL's public Hand/DrawPile structural check, inspect ReplayPrefix
    constraints, or depend on PendingChoice option order. It verifies only that the
    ordinary Training-visible decision remains usable and that the production Oracle/Beam
    path completes without replay_mismatch branch faults.
    """

    process, port = _start_paired_rl(pytest.skip)
    try:
        asyncio.run(_exercise_acrobatics_training_boundary(port))
    finally:
        _stop_paired_rl(process)
