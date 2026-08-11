from __future__ import annotations

from sts2_training.decision.policy import PriorHeuristicPolicy


def _action(action_id: str, option_id: str) -> dict:
    return {
        "action_id": action_id,
        "action_type": "choice_card",
        "is_available": True,
        "parameters": {"optionId": option_id},
    }


def _dto(operation: str, *, version: int = 1) -> dict:
    return {
        "pendingChoice": {
            "choiceSemantics": {"version": version, "operation": operation},
            "options": [
                {
                    "id": "GOOD",
                    "type": "Attack",
                    "rarity": "Rare",
                    "optionId": "option-good",
                },
                {
                    "id": "BAD",
                    "type": "Curse",
                    "optionId": "option-bad",
                },
            ],
            "selectedCount": 0,
            "selectedOptionIds": [],
        }
    }


def test_policy_prefers_low_quality_source_for_discard() -> None:
    actions = [_action("good", "option-good"), _action("bad", "option-bad")]

    ranked = PriorHeuristicPolicy().propose(actions, _dto("discard"), top_k=2)

    assert [candidate.action_id for candidate in ranked] == ["bad", "good"]


def test_policy_prefers_high_quality_card_for_retrieve() -> None:
    actions = [_action("bad", "option-bad"), _action("good", "option-good")]

    ranked = PriorHeuristicPolicy().propose(actions, _dto("retrieve"), top_k=2)

    assert [candidate.action_id for candidate in ranked] == ["good", "bad"]


def test_policy_keeps_future_semantics_neutral() -> None:
    actions = [_action("good", "option-good"), _action("bad", "option-bad")]

    ranked = PriorHeuristicPolicy().propose(
        actions,
        _dto("discard", version=2),
        top_k=2,
    )

    assert [candidate.action_id for candidate in ranked] == ["good", "bad"]


def test_policy_does_not_infer_from_legacy_or_incidental_fields() -> None:
    actions = [_action("good", "option-good"), _action("bad", "option-bad")]
    dto = _dto("unknown")
    pending = dto["pendingChoice"]
    pending.update(
        {
            "choiceOperation": "discard",
            "prompt": "Choose a card to discard",
            "selectorName": "DiscardSelector",
        }
    )

    ranked = PriorHeuristicPolicy().propose(actions, dto, top_k=2)

    assert [candidate.action_id for candidate in ranked] == ["good", "bad"]
