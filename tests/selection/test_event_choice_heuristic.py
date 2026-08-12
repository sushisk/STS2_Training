from sts2_training.selection.event_choice_heuristic import safe_event_option_candidates


def _event_action(action_id: str, will_kill_player) -> dict:
    return {
        "action_id": action_id,
        "action_type": "choice_event_option",
        "is_available": True,
        "parameters": {"eventId": "SLIPPERY_BRIDGE", "choiceId": action_id, "willKillPlayer": will_kill_player},
    }


def test_empty_without_event_option_candidates() -> None:
    assert safe_event_option_candidates([{"action_id": "a", "action_type": "card"}]) == []


def test_excludes_confirmed_lethal_option_when_a_safe_one_exists() -> None:
    candidates = [_event_action("overcome", None), _event_action("hold_on", True)]

    safe = safe_event_option_candidates(candidates)

    assert [a["action_id"] for a in safe] == ["overcome"]


def test_keeps_all_when_every_option_is_confirmed_lethal() -> None:
    candidates = [_event_action("a", True), _event_action("b", True)]

    safe = safe_event_option_candidates(candidates)

    assert {a["action_id"] for a in safe} == {"a", "b"}


def test_unevaluated_or_confirmed_safe_options_are_kept() -> None:
    candidates = [_event_action("unknown", None), _event_action("safe", False)]

    safe = safe_event_option_candidates(candidates)

    assert {a["action_id"] for a in safe} == {"unknown", "safe"}


def test_malformed_parameters_are_treated_as_not_confirmed_lethal() -> None:
    candidates = [
        {"action_id": "a", "action_type": "choice_event_option", "parameters": "not-a-dict"},
        {"action_id": "b", "action_type": "choice_event_option"},
    ]

    safe = safe_event_option_candidates(candidates)

    assert {a["action_id"] for a in safe} == {"a", "b"}
