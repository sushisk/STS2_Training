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
            "selectedCount": 1,
            "selectedOptionIds": ["option-0"],
            "options": [
                {"id": "STRIKE", "upgraded": False, "optionId": "option-1"},
                {"id": "STRIKE", "upgraded": True, "optionId": "option-2"},
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
    assert context.option_ids == ("option-1", "option-2")
    assert context.identity_valid is True


def test_missing_or_future_semantics_stay_neutral_and_are_not_inferred():
    dto = {
        "pendingChoice": {
            "choiceSemantics": {"version": 2, "operation": "discard"},
            "sourceEffectId": "card:LOOKS_LIKE_DISCARD",
            "selectedCount": 0,
            "selectedOptionIds": [],
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
    assert context.identity_valid is True


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


def test_choice_option_id_reads_only_canonical_parameters_identity():
    assert (
        choice_option_id(
            {
                "action_type": "choice_card",
                "parameters": {"optionId": "option-3", "cardId": "STRIKE"},
            }
        )
        == "option-3"
    )
    assert choice_option_id({"action_type": "choice_card", "optionId": "option-3"}) is None
    assert choice_option_id({"action_type": "card", "parameters": {"optionId": "option-3"}}) is None
    assert choice_option_id({"action_type": "choice_card", "parameters": {"cardId": "STRIKE"}}) is None
    assert choice_option_id({"action_type": "choice_card", "parameters": {"optionId": 3}}) is None


def test_choice_identity_marks_malformed_opaque_tokens_invalid():
    dto = {
        "pendingChoice": {
            "choiceSemantics": {"version": 1, "operation": "discard"},
            "sourceEffectId": "bad token with spaces",
            "selectedCount": 2,
            "selectedOptionIds": ["option-0", "bad token"],
            "options": [
                {"optionId": "option-1"},
                {"optionId": "bad token"},
            ],
        }
    }

    context = pending_choice_context(dto)

    assert context is not None
    assert context.source_effect_id is None
    assert context.selected_option_ids == ("option-0",)
    assert context.option_ids == ("option-1",)
    assert context.identity_valid is False
    assert (
        choice_option_id(
            {"action_type": "choice_card", "parameters": {"optionId": "bad token"}}
        )
        is None
    )


def test_choice_identity_detects_count_duplicate_and_overlap_inconsistency():
    base = {
        "choiceSemantics": {"version": 1, "operation": "discard"},
        "selectedCount": 1,
        "selectedOptionIds": ["option-0"],
        "options": [{"optionId": "option-1"}],
    }

    count_mismatch = {"pendingChoice": {**base, "selectedCount": 2}}
    context = pending_choice_context(count_mismatch)
    assert context is not None and context.identity_valid is False

    duplicate_selected = {
        "pendingChoice": {
            **base,
            "selectedCount": 2,
            "selectedOptionIds": ["option-0", "option-0"],
        }
    }
    context = pending_choice_context(duplicate_selected)
    assert context is not None and context.identity_valid is False

    overlap = {
        "pendingChoice": {
            **base,
            "options": [{"optionId": "option-0"}],
        }
    }
    context = pending_choice_context(overlap)
    assert context is not None and context.identity_valid is False
