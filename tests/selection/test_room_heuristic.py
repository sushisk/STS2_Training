from sts2_training.selection.room_heuristic import room_preference_scores


def _room_action(action_id: str, point_type: str) -> dict:
    return {
        "action_id": action_id,
        "action_type": "map_room",
        "is_available": True,
        "parameters": {"column": 0, "row": 0, "point_type": point_type},
    }


def test_room_preference_scores_empty_without_map_room_candidates() -> None:
    assert room_preference_scores([{"action_id": "a", "action_type": "card"}], {}) == {}


def test_treasure_and_rest_outrank_elite_at_full_hp() -> None:
    candidates = [
        _room_action("a-elite", "Elite"),
        _room_action("a-treasure", "Treasure"),
        _room_action("a-rest", "RestSite"),
        _room_action("a-monster", "Monster"),
    ]
    dto = {"hp": 80, "maxHp": 80}

    scores = room_preference_scores(candidates, dto)

    assert scores["a-treasure"] > scores["a-monster"]
    assert scores["a-rest"] > scores["a-monster"]
    assert scores["a-monster"] > scores["a-elite"]


def test_low_hp_strongly_prefers_rest_site_and_penalizes_elite() -> None:
    candidates = [_room_action("a-elite", "Elite"), _room_action("a-rest", "RestSite")]
    healthy_scores = room_preference_scores(candidates, {"hp": 80, "maxHp": 80})
    low_hp_scores = room_preference_scores(candidates, {"hp": 8, "maxHp": 80})

    assert low_hp_scores["a-rest"] > healthy_scores["a-rest"]
    assert low_hp_scores["a-elite"] < healthy_scores["a-elite"]
    assert low_hp_scores["a-rest"] > low_hp_scores["a-elite"]


def test_missing_hp_data_degrades_to_point_type_only_score() -> None:
    candidates = [_room_action("a-rest", "RestSite")]
    scores = room_preference_scores(candidates, {})
    assert scores["a-rest"] == 5.0


def test_unknown_point_type_is_neutral() -> None:
    candidates = [_room_action("a-mystery", "SomeFuturePointType")]
    scores = room_preference_scores(candidates, {"hp": 80, "maxHp": 80})
    assert scores["a-mystery"] == 0.0
