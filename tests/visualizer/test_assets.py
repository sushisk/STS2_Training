from sts2_training.visualizer.assets import INDEX_HTML


def test_browser_uses_canonical_power_contract() -> None:
    assert "value.powers" in INDEX_HTML
    assert "renderPowers" in INDEX_HTML
    assert "power-strip" in INDEX_HTML
    assert "player_state" not in INDEX_HTML
    assert "power_list" not in INDEX_HTML
    assert "active_powers" not in INDEX_HTML
