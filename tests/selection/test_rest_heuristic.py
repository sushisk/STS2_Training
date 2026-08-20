from sts2_training.selection.rest_heuristic import (
    rest_option_preference_scores,
    rest_option_quality_score,
)


def _rest_action(action_id: str, option_id: str) -> dict:
    return {
        "action_id": action_id,
        "action_type": "choice_rest_option",
        "label": option_id,
        "is_available": True,
        "parameters": {"restOptionId": option_id},
    }


def _dto(hp: int, max_hp: int = 80) -> dict:
    return {"hp": hp, "maxHp": max_hp}


def _best(candidates: list[dict], dto: dict) -> str:
    scores = rest_option_preference_scores(candidates, dto)
    return max(scores, key=lambda key: scores[key])


def test_no_scores_without_rest_option_candidates() -> None:
    assert rest_option_preference_scores([{"action_id": "a", "action_type": "card"}], {}) == {}


def test_low_hp_prefers_healing() -> None:
    """The case this module exists for: a 5-run evaluation upgraded at 8, 18 and 30 HP."""

    candidates = [_rest_action("a-heal", "HEAL"), _rest_action("a-smith", "SMITH")]

    for hp in (8, 18, 30, 40):
        assert _best(candidates, _dto(hp)) == "a-heal", hp


def test_high_hp_prefers_the_permanent_upgrade() -> None:
    candidates = [_rest_action("a-heal", "HEAL"), _rest_action("a-smith", "SMITH")]

    for hp in (60, 70, 72, 80):
        assert _best(candidates, _dto(hp)) == "a-smith", hp


def test_crossover_is_at_seventy_percent_hp() -> None:
    candidates = [_rest_action("a-heal", "HEAL"), _rest_action("a-smith", "SMITH")]

    assert _best(candidates, _dto(55)) == "a-heal"
    assert _best(candidates, _dto(57)) == "a-smith"


def test_mend_is_treated_as_healing() -> None:
    candidates = [_rest_action("a-mend", "MEND"), _rest_action("a-smith", "SMITH")]

    assert _best(candidates, _dto(10)) == "a-mend"


def test_healing_score_grows_as_hp_drops() -> None:
    scores = [rest_option_quality_score("HEAL", hp / 80.0) for hp in (80, 60, 40, 20, 0)]

    assert scores == sorted(scores)
    assert scores[0] == 0.0


def test_unreadable_hp_falls_back_to_healing() -> None:
    candidates = [_rest_action("a-heal", "HEAL"), _rest_action("a-smith", "SMITH")]

    for dto in ({}, {"hp": None, "maxHp": 80}, {"hp": 40, "maxHp": 0}):
        assert _best(candidates, dto) == "a-heal", dto


def test_unrecognized_option_is_neutral_and_never_beats_a_needed_heal() -> None:
    candidates = [
        _rest_action("a-heal", "HEAL"),
        _rest_action("a-dig", "DIG"),
        _rest_action("a-smith", "SMITH"),
    ]

    assert rest_option_quality_score("DIG", 0.5) == 0.0
    assert _best(candidates, _dto(8)) == "a-heal"
    assert _best(candidates, _dto(80)) == "a-smith"


def test_option_id_falls_back_to_the_label() -> None:
    action = {
        "action_id": "a-heal",
        "action_type": "choice_rest_option",
        "label": "HEAL",
        "is_available": True,
    }

    scores = rest_option_preference_scores([action], _dto(8))

    assert scores["a-heal"] > 0.0
