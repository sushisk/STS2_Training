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


def _card_action(
    action_id: str, card_id: str, *, cost: int = 1, target_type: str = "Self"
) -> dict:
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
    def test_policy_only_ranks_without_structural_injection(self) -> None:
        actions = [
            _action("end", "system"),
            _action("card-1", "card"),
            _action("card-2", "card"),
        ]

        candidates = PriorHeuristicPolicy().propose(actions, {}, top_k=2)

        self.assertEqual([candidate.action_id for candidate in candidates], ["card-1", "card-2"])
        self.assertTrue(all(isinstance(candidate, ActionCandidate) for candidate in candidates))

    def test_excludes_unavailable_actions(self) -> None:
        candidates = PriorHeuristicPolicy().propose(
            [
                _action("a", "card", is_available=False),
                _action("b", "card"),
            ],
            {},
            top_k=3,
        )
        self.assertEqual([candidate.action_id for candidate in candidates], ["b"])

    def test_availability_is_checked_once_per_action(self) -> None:
        actions = [
            _CountingAction(_action("a", "card")),
            _CountingAction(_action("b", "card", is_available=False)),
        ]

        PriorHeuristicPolicy().propose(actions, {}, top_k=3)

        self.assertEqual([action.availability_reads for action in actions], [1, 1])

    def test_empty_when_no_available_actions(self) -> None:
        self.assertEqual(
            PriorHeuristicPolicy().propose(
                [_action("a", "card", is_available=False)], {}, top_k=2
            ),
            [],
        )

    def test_top_k_truncates(self) -> None:
        actions = [_action(str(index), "card") for index in range(5)]
        self.assertEqual(len(PriorHeuristicPolicy().propose(actions, {}, top_k=2)), 2)

    def test_rejects_non_positive_top_k(self) -> None:
        with self.assertRaises(ValueError):
            PriorHeuristicPolicy().propose([_action("a", "card")], {}, top_k=0)

    def test_rng_tie_break_is_deterministic_for_seeded_rng(self) -> None:
        actions = [_action(str(index), "card") for index in range(10)]
        result_a = [
            candidate.action_id
            for candidate in PriorHeuristicPolicy(rng=random.Random(42)).propose(
                actions, {}, top_k=10
            )
        ]
        result_b = [
            candidate.action_id
            for candidate in PriorHeuristicPolicy(rng=random.Random(42)).propose(
                actions, {}, top_k=10
            )
        ]
        self.assertEqual(result_a, result_b)

    def test_propose_batch_default_matches_looping_propose(self) -> None:
        actions = [_action("a", "card")]
        policy = PriorHeuristicPolicy()
        batched = policy.propose_batch([(actions, {}), (actions, {})], top_k=1)
        self.assertEqual(batched, [policy.propose(actions, {}, top_k=1)] * 2)

    def test_lethal_incoming_damage_prioritizes_defensive_skill(self) -> None:
        actions = [
            _card_action("strike", "STRIKE", target_type="SingleEnemy"),
            _card_action("defend", "DEFEND_IRONCLAD"),
            _card_action("power", "SCALING_POWER"),
        ]
        dto = {
            "hp": 12,
            "block": 0,
            "energy": 3,
            "hand": [
                {"id": "STRIKE", "type": "Attack", "cost": 1},
                {"id": "DEFEND_IRONCLAD", "type": "Skill", "cost": 1},
                {"id": "SCALING_POWER", "type": "Power", "rarity": "Rare", "cost": 1},
            ],
            "enemies": [
                {"isAlive": True, "intent": {"attackDamage": 15, "attackRepeats": 1}}
            ],
        }

        candidates = PriorHeuristicPolicy().propose(actions, dto, top_k=3)

        self.assertEqual(candidates[0].action_id, "defend")

    def test_safe_turn_prefers_scaling_power_over_basic_attack(self) -> None:
        actions = [
            _card_action("strike", "STRIKE", target_type="SingleEnemy"),
            _card_action("power", "SCALING_POWER"),
        ]
        dto = {
            "hp": 70,
            "energy": 3,
            "hand": [
                {"id": "STRIKE", "type": "Attack", "cost": 1},
                {"id": "SCALING_POWER", "type": "Power", "rarity": "Rare", "cost": 1},
            ],
            "enemies": [{"isAlive": True, "intent": {}}],
        }
        self.assertEqual(
            PriorHeuristicPolicy().propose(actions, dto, top_k=1)[0].action_id,
            "power",
        )

    def test_lethal_pressure_promotes_potion(self) -> None:
        actions = [
            _card_action("strike", "STRIKE", target_type="SingleEnemy"),
            _action("potion", "potion", parameters={"potionSlot": 0, "targetType": "Self"}),
        ]
        dto = {
            "hp": 10,
            "energy": 3,
            "hand": [{"id": "STRIKE", "type": "Attack", "cost": 1}],
            "potions": [{"rarity": "Common"}],
            "enemies": [{"isAlive": True, "intent": {"attackDamage": 20}}],
        }

        candidates = PriorHeuristicPolicy().propose(actions, dto, top_k=2)

        self.assertEqual(candidates[0].action_id, "potion")

    def test_choice_target_prefers_low_hp_dangerous_enemy(self) -> None:
        actions = [
            _action(
                "safe",
                "choice_target",
                parameters={"enemyIndex": 0, "hp": 35, "maxHp": 40, "block": 0},
            ),
            _action(
                "dangerous",
                "choice_target",
                parameters={"enemyIndex": 1, "hp": 5, "maxHp": 40, "block": 0},
            ),
        ]
        dto = {
            "enemies": [
                {"index": 0, "isAlive": True, "intent": {"attackDamage": 5}},
                {"index": 1, "isAlive": True, "intent": {"attackDamage": 18}},
            ]
        }
        self.assertEqual(
            PriorHeuristicPolicy().propose(actions, dto, top_k=1)[0].action_id,
            "dangerous",
        )

    def test_choice_card_stays_neutral_even_with_incidental_semantics_keys(self) -> None:
        actions = [
            _action("curse", "choice_card", label="CURSE"),
            _action("rare", "choice_card", label="RARE"),
        ]
        options = [
            {"id": "CURSE", "type": "Curse", "rarity": "Curse", "cost": 0},
            {"id": "RARE", "type": "Attack", "rarity": "Rare", "cost": 2},
        ]
        for pending_extra in (
            {},
            {"semantics": "gain"},
            {"choiceSemantics": "gain"},
            {"operation": "discard"},
            {"choiceType": "upgrade"},
            {"kind": "exhaust"},
        ):
            with self.subTest(pending_extra=pending_extra):
                dto = {"pendingChoice": {"options": options, **pending_extra}}
                candidates = PriorHeuristicPolicy().propose(actions, dto, top_k=2)
                self.assertEqual(
                    [candidate.action_id for candidate in candidates],
                    ["curse", "rare"],
                )

    def test_confirm_scores_above_choice_after_maximum_selection(self) -> None:
        actions = [
            _action("card", "choice_card"),
            _action("confirm", "choice_confirm"),
        ]
        dto = {
            "pendingChoice": {"selectedCount": 1, "minSelect": 1, "maxSelect": 1}
        }
        self.assertEqual(
            PriorHeuristicPolicy().propose(actions, dto, top_k=1)[0].action_id,
            "confirm",
        )

    def test_ambiguous_duplicate_does_not_inherit_one_copy_upgrade(self) -> None:
        actions = [
            _card_action("dup", "DUP", cost=1),
            _card_action("other", "OTHER", cost=1),
        ]
        dto = {
            "energy": 3,
            "hand": [
                {"id": "DUP", "type": "Attack", "cost": 1, "upgraded": True},
                {"id": "DUP", "type": "Attack", "cost": 1, "upgraded": False},
                {"id": "OTHER", "type": "Attack", "cost": 1, "upgraded": False},
            ],
        }

        candidates = PriorHeuristicPolicy().propose(actions, dto, top_k=2)

        self.assertEqual([candidate.action_id for candidate in candidates], ["dup", "other"])


if __name__ == "__main__":
    unittest.main()
