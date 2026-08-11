from sts2_training.selection.choice_semantics import (
    choice_option_id,
    parse_choice_semantics,
    pending_choice_context,
)


def test_pending_choice_context_consumes_canonical_semantics_and_option_identity():
    dto = {
        "pendingChoice": {
            "choiceSemantics": {
                "version": 1,
                "operation": "discard",
                "effect": "move",
                "sourceZone": "hand",
                "destinationZone": "discard_pile",
                "orderMatters": False,
                "replacementAllowed": False,
            },
            "sourceEffectId": "card:SURVIVOR",
            "selectedOptionIds": ["option-0"],
            "options": [
                {"id": "STRIKE", "upgraded": False, "optionId": "option-0"},
                {"id": "STRIKE", "upgraded": False, "optionId": "option-1"},
            ],
        }
    }

    context = pending_choice_context(dto)

    assert context is not None
    assert context.semantics.operation == "discard"
    assert context.semantics.effect == "move"
    assert context.semantics.source_zone == "hand"
    assert context.semantics.destination_zone == "discard_pile"
    assert context.source_effect_id == "card:SURVIVOR"
    assert context.selected_option_ids == ("option-0",)
    assert context.option_ids == ("option-0", "option-1")


def test_missing_or_future_semantics_stay_neutral_and_are_not_inferred():
    dto = {
        "pendingChoice": {
            "choiceSemantics": {"version": 2, "operation": "discard"},
            "sourceEffectId": "card:LOOKS_LIKE_DISCARD",
            "prompt": "Choose a card to discard",
            "selectorName": "DiscardSelector",
            "options": [{"id": "SURVIVOR", "optionId": "option-0"}],
        }
    }

    context = pending_choice_context(dto)

    assert context is not None
    assert context.semantics.operation == "unknown"
    assert context.source_effect_id is None
    assert context.option_ids == ("option-0",)


def test_malformed_v1_factor_degrades_to_unknown():
    semantics = parse_choice_semantics(
        {
            "version": 1,
            "operation": "upgrade",
            "modifier": "future_modifier",
        }
    )

    assert semantics.operation == "unknown"
    assert semantics.is_known is False


def test_choice_option_id_reads_only_explicit_opaque_identity():
    assert choice_option_id({"action_type": "choice_card", "optionId": "option-3"}) == "option-3"
    assert choice_option_id({"action_type": "choice_card", "cardId": "STRIKE"}) is None
    assert choice_option_id({"action_type": "choice_card", "optionId": 3}) is None
