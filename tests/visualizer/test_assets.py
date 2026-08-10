from sts2_training.visualizer.assets import INDEX_HTML


def test_browser_uses_canonical_power_contract() -> None:
    assert "value.powers" in INDEX_HTML
    assert "renderPowers" in INDEX_HTML
    assert "power-strip" in INDEX_HTML
    assert "player_state" not in INDEX_HTML
    assert "power_list" not in INDEX_HTML
    assert "active_powers" not in INDEX_HTML


def test_browser_formats_hp_and_choices_from_canonical_contract() -> None:
    assert "const hpText" in INDEX_HTML
    assert "renderChoices(frame.choices,event)" in INDEX_HTML
    assert "choice.details" in INDEX_HTML
    assert "choice.summary" in INDEX_HTML
    assert "choices-zone" in INDEX_HTML
    assert "Object.entries(action)" not in INDEX_HTML
