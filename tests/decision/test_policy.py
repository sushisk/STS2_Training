from __future__ import annotations

import random
import unittest

from sts2_training.decision.policy import ActionCandidate, PriorHeuristicPolicy


def _action(
    action_id: str,
    action_type: str,
    *,
    is_available: bool = True,
    label: str | None = None,
    parameters: dict | None = None,
) -> dict:
    return {
        "action_id": action_id,
        "action_type": action_type,
        "is_available": is_available,
        "label": label or action_id,
        "parameters": parameters or {},
    }


def _card_action(action_id: str, card_id: str, *, cost: int = 1, target_type: str = "Self") -> dict:
    return _action(
        action_id,
        "card",
        label=card_id,
        parameters={"cardId": card_id, "cost": cost, "targetType": target_type},
    )


class _CountingAction(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.availability_reads = 0

    def get(self, key, default=None):
        if key == "is_available":
            self.availability_reads += 1
        return super().get(key, default)


class PriorHeuristicPolicyTest(unittest.TestCase):
    def test_keeps_end_turn_in_small_combat_shortlist(self) -> None:
        legal_actions = [
            _action("a-end", "system"),
            _action("a-card1", "card"),
            _action("a-card2", "card"),
            _action("a-choice", "choice_card"),
        ]
        policy = PriorHeuristicPolicy()

        candidates = policy.propose(legal_actions, {}, top_k=2)

        self.assertEqual([c.action_id for c in candidates], ["a-card1", "a-end"])
        self.assertTrue(all(isinstance(c, ActionCandidate) for c in candidates))

    def test_top_one_still_uses_best_ranked_action(self) -> None:
        legal_actions = [_action("a-end", "system"), _action("a-card", "card")]
        policy = PriorHeuristicPolicy()

        candidates = policy.propose(legal_actions, {}, top_k=1)

        self.assertEqual([c.action_id for c in candidates], ["a-card"])

    def test_falls_back_to_other_categories_when_none_of_priority_present(self) -> None:
        legal_actions = [_action("a-end", "system"), _action("a-potion", "potion")]
        policy = PriorHeuristicPolicy()

        candidates = policy.propose(legal_actions, {}, top_k=5)

        self.assertEqual({c.action_id for c in candidates}, {"a-end", "a-potion"})

    def test_excludes_unavailable_actions(self) -> None:
        legal_actions = [
            _action("a-card1", "card", is_available=False),
            _action("a-card2", "card"),
        ]
        policy = PriorHeuristicPolicy()

        candidates = policy.propose(legal_actions, {}, top_k=5)

        self.assertEqual([c.action_id for c in candidates], ["a-card2"])

    def test_availability_is_checked_once_per_action(self) -> None:
        actions = [
            _CountingAction(_action("a-card1", "card")),
            _CountingAction(_action("a-card2", "card", is_available=False)),
        ]
        policy = PriorHeuristicPolicy()

        policy.propose(actions, {}, top_k=5)

        self.assertEqual([action.availability_reads for action in actions], [1, 1])

    def test_empty_when_no_available_actions(self) -> None:
        legal_actions = [_action("a-card1", "card", is_available=False)]
        policy = PriorHeuristicPolicy()

        self.assertEqual(policy.propose(legal_actions, {}, top_k=5), [])

    def test_top_k_truncates(self) -> None:
        legal_actions = [_action(f"a-card{i}", "card") for i in range(5)]
        policy = PriorHeuristicPolicy()

        candidates = policy.propose(legal_actions, {}, top_k=2)

        self.assertEqual(len(candidates), 2)

    def test_rejects_non_positive_top_k(self) -> None:
        policy = PriorHeuristicPolicy()
        with self.assertRaises(ValueError):
            policy.propose([_action("a", "card")], {}, top_k=0)

    def test_rng_tie_break_is_deterministic_for_seeded_rng(self) -> None:
        legal_actions = [_action(f"a-card{i}", "card") for i in range(10)]
        policy_a = PriorHeuristicPolicy(rng=random.Random(42))
        policy_b = PriorHeuristicPolicy(rng=random.Random(42))

        result_a = [c.action_id for c in policy_a.propose(legal_actions, {}, top_k=10)]
        result_b = [c.action_id for c in policy_b.propose(legal_actions, {}, top_k=10)]

        self.assertEqual(result_a, result_b)

    def test_propose_batch_default_matches_looping_propose(self) -> None:
        legal_actions = [_action("a-card1", "card")]
        policy = PriorHeuristicPolicy()

        batched = policy.propose_batch([(legal_actions, {}), (legal_actions, {})], top_k=1)

        self.assertEqual(len(batched), 2)
        self.assertEqual(batched[0], policy.propose(legal_actions, {}, top_k=1))

    def test_lethal_incoming_damage_prioritizes_defensive_skill_over_attack_and_power(self) -> None:
        legal_actions = [
            _action("0", "system"),
            _card_action("1", "STRIKE_IRONCLAD", target_type="SingleEnemy"),
            _card_action("2", "DEFEND_IRONCLAD"),
            _card_action("3", "SCALING_POWER"),
        ]
        dto = {
            "hp": 12,
            "maxHp": 80,
            "block": 0,
            "energy": 3,
            "hand": [
                {"id": "STRIKE_IRONCLAD", "type": "Attack", "rarity": "Basic", "cost": 1},
                {"id": "DEFEND_IRONCLAD", "type": "Skill", "rarity": "Basic", "cost": 1},
                {"id": "SCALING_POWER", "type": "Power", "rarity": "Rare", "cost": 1},
            ],
            "enemies": [
                {
                    "index": 0,
                    "hp": 40,
                    "maxHp": 40,
                    "isAlive": True,
                    "intent": {"attackDamage": 15, "attackRepeats": 1},
                }
            ],
        }
        policy = PriorHeuristicPolicy()

        candidates = policy.propose(legal_actions, dto, top_k=3)

        self.assertEqual(candidates[0].action_id, "2")
        self.assertIn("0", [c.action_id for c in candidates])

    def test_safe_turn_prefers_scaling_power_over_basic_attack(self) -> None:
        legal_actions = [
            _card_action("1", "STRIKE_IRONCLAD", target_type="SingleEnemy"),
            _card_action("2", "SCALING_POWER"),
        ]
        dto = {
            "hp": 70,
            "maxHp": 80,
            "energy": 3,
            "hand": [
                {"id": "STRIKE_IRONCLAD", "type": "Attack", "rarity": "Basic", "cost": 1},
                {"id": "SCALING_POWER", "type": "Power", "rarity": "Rare", "cost": 1},
            ],
            "enemies": [{"index": 0, "hp": 50, "maxHp": 50, "isAlive": True, "intent": {}}],
        }
        policy = PriorHeuristicPolicy()

        candidates = policy.propose(legal_actions, dto, top_k=1)

        self.assertEqual(candidates[0].action_id, "2")

    def test_lethal_pressure_promotes_potion_into_top_candidates(self) -> None:
        legal_actions = [
            _action("0", "system"),
            _card_action("1", "STRIKE_A", target_type="SingleEnemy"),
            _card_action("2", "STRIKE_B", target_type="SingleEnemy"),
            _action(
                "3",
                "potion",
                label="BLOCK_POTION",
                parameters={"potionId": "BLOCK_POTION", "potionSlot": 0, "targetType": "Self"},
            ),
        ]
        dto = {
            "hp": 10,
            "maxHp": 80,
            "energy": 3,
            "hand": [
                {"id": "STRIKE_A", "type": "Attack", "cost": 1},
                {"id": "STRIKE_B", "type": "Attack", "cost": 1},
            ],
            "potions": [{"id": "BLOCK_POTION", "rarity": "Common", "targetType": "Self"}],
            "enemies": [
                {
                    "index": 0,
                    "hp": 50,
                    "maxHp": 50,
                    "isAlive": True,
                    "intent": {"attackDamage": 20},
                }
            ],
        }
        policy = PriorHeuristicPolicy()

        candidates = policy.propose(legal_actions, dto, top_k=3)

        self.assertIn("3", [c.action_id for c in candidates])
        self.assertIn("0", [c.action_id for c in candidates])

    def test_lethal_pressure_preserves_a_card_branch_with_many_potions(self) -> None:
        legal_actions = [
            _action("end", "system"),
            _card_action("card", "STRIKE", target_type="SingleEnemy"),
            *[
                _action(
                    f"p{i}",
                    "potion",
                    parameters={"potionSlot": i, "targetType": "Self"},
                )
                for i in range(3)
            ],
        ]
        dto = {
            "hp": 10,
            "energy": 3,
            "hand": [{"id": "STRIKE", "type": "Attack", "cost": 1}],
            "potions": [{"rarity": "Common"} for _ in range(3)],
            "enemies": [{"isAlive": True, "intent": {"attackDamage": 20}}],
        }

        candidates = PriorHeuristicPolicy().propose(legal_actions, dto, top_k=4)
        ids = [candidate.action_id for candidate in candidates]

        self.assertIn("card", ids)
        self.assertIn("end", ids)

    def test_choice_target_prefers_low_hp_enemy_with_dangerous_intent(self) -> None:
        legal_actions = [
            _action(
                "10",
                "choice_target",
                parameters={"enemyIndex": 0, "hp": 35, "maxHp": 40, "block": 0},
            ),
            _action(
                "11",
                "choice_target",
                parameters={"enemyIndex": 1, "hp": 5, "maxHp": 40, "block": 0},
            ),
        ]
        dto = {
            "enemies": [
                {
                    "index": 0,
                    "hp": 35,
                    "maxHp": 40,
                    "isAlive": True,
                    "intent": {"attackDamage": 5},
                },
                {
                    "index": 1,
                    "hp": 5,
                    "maxHp": 40,
                    "isAlive": True,
                    "intent": {"attackDamage": 18},
                },
            ]
        }
        policy = PriorHeuristicPolicy()

        candidates = policy.propose(legal_actions, dto, top_k=1)

        self.assertEqual(candidates[0].action_id, "11")

    def test_choice_card_uses_pending_choice_card_quality_when_semantics_are_known(self) -> None:
        legal_actions = [
            _action("20", "choice_card", label="CURSE_A"),
            _action("21", "choice_card", label="RARE_ATTACK"),
            _action("22", "choice_confirm"),
        ]
        dto = {
            "pendingChoice": {
                "semantics": "gain",
                "selectedCount": 0,
                "minSelect": 1,
                "maxSelect": 1,
                "options": [
                    {"id": "CURSE_A", "type": "Curse", "rarity": "Curse", "cost": 0},
                    {"id": "RARE_ATTACK", "type": "Attack", "rarity": "Rare", "cost": 2},
                ],
            }
        }
        policy = PriorHeuristicPolicy()

        candidates = policy.propose(legal_actions, dto, top_k=1)

        self.assertEqual(candidates[0].action_id, "21")

    def test_choice_card_is_neutral_when_semantics_are_unknown(self) -> None:
        legal_actions = [
            _action("20", "choice_card", label="CURSE_A"),
            _action("21", "choice_card", label="RARE_ATTACK"),
        ]
        dto = {
            "pendingChoice": {
                "selectedCount": 0,
                "minSelect": 1,
                "maxSelect": 1,
                "options": [
                    {"id": "CURSE_A", "type": "Curse", "rarity": "Curse", "cost": 0},
                    {"id": "RARE_ATTACK", "type": "Attack", "rarity": "Rare", "cost": 2},
                ],
            }
        }

        candidates = PriorHeuristicPolicy().propose(legal_actions, dto, top_k=2)

        self.assertEqual([candidate.action_id for candidate in candidates], ["20", "21"])

    def test_discard_semantics_reverse_card_quality(self) -> None:
        legal_actions = [
            _action("20", "choice_card", label="CURSE_A"),
            _action("21", "choice_card", label="RARE_ATTACK"),
        ]
        dto = {
            "pendingChoice": {
                "operation": "discard",
                "options": [
                    {"id": "CURSE_A", "type": "Curse", "rarity": "Curse", "cost": 0},
                    {"id": "RARE_ATTACK", "type": "Attack", "rarity": "Rare", "cost": 2},
                ],
            }
        }

        candidates = PriorHeuristicPolicy().propose(legal_actions, dto, top_k=1)

        self.assertEqual(candidates[0].action_id, "20")

    def test_confirm_branch_is_retained_before_maximum_selection(self) -> None:
        legal_actions = [
            *[_action(str(i), "choice_card", label=f"CARD_{i}") for i in range(5)],
            _action("confirm", "choice_confirm"),
        ]
        dto = {
            "pendingChoice": {
                "semantics": "gain",
                "selectedCount": 1,
                "minSelect": 1,
                "maxSelect": 5,
                "options": [
                    {"id": f"CARD_{i}", "type": "Attack", "rarity": "Rare", "cost": 0}
                    for i in range(5)
                ],
            }
        }

        candidates = PriorHeuristicPolicy().propose(legal_actions, dto, top_k=4)

        self.assertIn("confirm", [candidate.action_id for candidate in candidates])

    def test_ambiguous_duplicate_cards_do_not_inherit_one_copys_upgrade_bonus(self) -> None:
        legal_actions = [
            _card_action("a", "DUP", cost=1),
            _card_action("b", "OTHER", cost=1),
        ]
        dto = {
            "energy": 3,
            "hand": [
                {"id": "DUP", "type": "Attack", "rarity": "Common", "cost": 1, "upgraded": True},
                {"id": "DUP", "type": "Attack", "rarity": "Common", "cost": 1, "upgraded": False},
                {"id": "OTHER", "type": "Attack", "rarity": "Common", "cost": 1, "upgraded": False},
            ],
        }

        candidates = PriorHeuristicPolicy().propose(legal_actions, dto, top_k=2)

        self.assertEqual([candidate.action_id for candidate in candidates], ["a", "b"])

    def test_confirm_is_prioritized_after_maximum_choice_count(self) -> None:
        legal_actions = [
            _action("20", "choice_card", label="ATTACK"),
            _action("22", "choice_confirm"),
            _action("23", "choice_skip"),
        ]
        dto = {
            "pendingChoice": {
                "selectedCount": 1,
                "minSelect": 1,
                "maxSelect": 1,
                "options": [{"id": "ATTACK", "type": "Attack", "rarity": "Common", "cost": 1}],
            }
        }
        policy = PriorHeuristicPolicy()

        candidates = policy.propose(legal_actions, dto, top_k=1)

        self.assertEqual(candidates[0].action_id, "22")


if __name__ == "__main__":
    unittest.main()
