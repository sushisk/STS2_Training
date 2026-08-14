import pytest
from sts2_training.runner.combat_resource_reward import (
    CombatResourceSnapshot, combat_resource_quality, combat_resource_snapshot,
)

def test_resource_quality_prefers_hp_and_potion_retention():
    strong = CombatResourceSnapshot(80.0, 100.0, 2, 2)
    weak = CombatResourceSnapshot(10.0, 100.0, 0, 2)
    assert combat_resource_quality(strong) == pytest.approx(0.84)
    assert combat_resource_quality(weak) == pytest.approx(0.08)

def test_no_initial_potions_is_neutral_not_a_penalty():
    snapshot = CombatResourceSnapshot(50.0, 100.0, 0, 0)
    assert combat_resource_quality(snapshot) == pytest.approx(0.6)

def test_terminal_dto_snapshot_counts_present_potions():
    dto = {"hp": 30, "maxHp": 40, "potions": [{"slot": 0, "potion_id": "A"}]}
    snapshot = combat_resource_snapshot(dto, initial_potion_count=2)
    assert snapshot == CombatResourceSnapshot(30.0, 40.0, 1, 2)

def test_terminal_dto_snapshot_counts_sparse_potion_slots():
    dto = {"hp": 30, "maxHp": 40, "potions": [{"slot": 0, "potion_id": "A"}, None, None]}
    snapshot = combat_resource_snapshot(dto, initial_potion_count=3)
    assert snapshot == CombatResourceSnapshot(30.0, 40.0, 1, 3)

def test_terminal_dto_snapshot_counts_no_present_potions():
    dto = {"hp": 30, "maxHp": 40, "potions": [None, None, None]}
    snapshot = combat_resource_snapshot(dto, initial_potion_count=3)
    assert snapshot == CombatResourceSnapshot(30.0, 40.0, 0, 3)

def test_terminal_dto_requires_hp_contract():
    with pytest.raises(ValueError):
        combat_resource_snapshot({"maxHp": 40, "potions": []}, initial_potion_count=0)
