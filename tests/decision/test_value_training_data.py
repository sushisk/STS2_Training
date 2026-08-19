from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts2_training.api.contract import SCHEMA_VERSION
from sts2_training.decision.oracle_log import (
    ORACLE_EPISODE_RESULT_SCHEMA_VERSION,
    ORACLE_RECORD_SCHEMA_VERSION,
)
from sts2_training.decision.value_training_data import (
    load_combat_value_examples,
    load_combat_value_rl_episodes,
)
from tests.dto_test_helpers import action, card, dto, dto_get, dto_replace, enemy, intent


_DTO_VERSION = "emulator-test"


def _provenance() -> dict:
    return {
        "training_commit": "abc",
        "teacher_policy_class": "teacher.Policy",
        "teacher_inner_policy_class": "teacher.Policy",
        "teacher_coverage_policy_class": None,
        "teacher_value_class": "teacher.Value",
        "teacher_policy_metadata": {},
        "teacher_inner_policy_metadata": {},
        "teacher_value_metadata": {},
        "pruner_name": "value_top_k",
        "pruner_version": "1",
        "rng_sampling": "independent",
    }


def _target_metadata() -> dict:
    return {
        "oracle_beam_width": 32,
        "target_beam_width": 8,
        "top_k_actions": 8,
        "max_depth": 4,
        "max_continuation_steps": 8,
        "time_budget_ms": 5000.0,
        "exhaustive_root_actions": True,
        "pruner_name": "value_top_k",
        "pruner_version": "1",
        "rng_sampling": "independent",
    }


def _contract(dto_version: str = _DTO_VERSION) -> dict:
    return {
        "wire_schema_version": SCHEMA_VERSION,
        "mask_version": "1.2",
        "dto_version": dto_version,
    }


def _card(card_id: str, type_: str, *, upgrade_level: int = 0, enchantment=None, count=None):
    fields = dict(
        id=card_id,
        type=type_,
        upgraded=upgrade_level > 0,
        upgrade_level=upgrade_level,
        tinker_time_type=None,
        tinker_time_rider=None,
        enchantment=enchantment,
    )
    if count is not None:
        fields["count"] = count
    return card(**fields)


def _dto(*, dto_version: str = _DTO_VERSION, terminal: bool = False) -> dict:
    state = dto(
        mask_version="1.2",
        dto_version=dto_version,
        hp=40,
        max_hp=80,
        block=0,
        energy=3,
        enemies=[
            enemy(
                hp=20,
                max_hp=40,
                is_alive=True,
                intent=intent(attack_damage=5, attack_repeats=1),
                powers=[],
            )
        ],
        hand=[_card("STRIKE", "Attack", upgrade_level=1)],
        draw_pile=[
            _card(
                "DEFEND",
                "Skill",
                enchantment={"id": "SHARP", "amount": 2, "status": "Normal"},
                count=2,
            )
        ],
        discard_pile=[],
        exhaust_pile=[],
        potions=[],
        player_powers=[],
        legal_actions=[],
    )
    if terminal:
        state = dto_replace(state, terminal=True, outcome="victory")
    return state


def _sample(action_id: str, rng_id: int, *, target_value, target_source: str) -> dict:
    return {
        "action_id": action_id,
        "action": action(id=action_id, type="card"),
        "rng_id": rng_id,
        "root_state_node_id": f"n-{action_id}",
        "decision_point_id": f"after-{action_id}",
        "masked_emulator_dto": _dto(),
        "target_value": target_value,
        "target_source": target_source,
        "terminal_reached": target_source == "terminal",
        "deepest_combat_depth": 3,
        "censored": target_source != "terminal",
        "censor_reason": None if target_source == "terminal" else "time_budget",
        "best_node_id": None if target_source == "no_target" else f"leaf-{action_id}",
    }


def _record(*samples: dict, dto_version: str = _DTO_VERSION, decision_index: int = 0) -> dict:
    root_dto = _dto(dto_version=dto_version)
    next_dto = _dto(dto_version=dto_version)
    for sample in samples:
        sample.setdefault("masked_emulator_dto", _dto(dto_version=dto_version))
    return {
        "record_type": "combat_oracle_decision",
        "record_schema_version": ORACLE_RECORD_SCHEMA_VERSION,
        "instance_id": "inst-1",
        "decision_index": decision_index,
        "decision_point_id": f"d{decision_index}",
        "dto_contract": _contract(dto_version),
        "decision_response_metadata": {
            "decision_point_id": f"d{decision_index}",
            "server_epoch": "epoch-1",
        },
        "masked_emulator_dto": root_dto,
        "root_value_samples": list(samples),
        "runtime_transition": {
            "chosen_action_id": "actual",
            "chosen_action": action(id="actual", type="card"),
            "next_decision_point_id": f"d{decision_index + 1}",
            "commit_response_metadata": {
                "decision_point_id": f"d{decision_index + 1}",
                "server_epoch": "epoch-1",
            },
            "next_masked_emulator_dto": next_dto,
            "next_dto_contract": _contract(dto_version),
            "combat_result": None,
        },
        "oracle_targets": {"metadata": _target_metadata()},
        "provenance": _provenance(),
    }


def _episode(*, final_dto: dict, decisions_collected: int, completed: bool, combat_result) -> dict:
    return {
        "record_type": "combat_oracle_episode_result",
        "record_schema_version": ORACLE_EPISODE_RESULT_SCHEMA_VERSION,
        "instance_id": "inst-1",
        "decisions_collected": decisions_collected,
        "completed": completed,
        "termination_reason": "terminal" if completed else "max_decisions",
        "combat_result": combat_result,
        "dto_contract": _contract(dto_get(final_dto, "dto_version")),
        "final_decision_metadata": {
            "decision_point_id": f"d{decisions_collected}",
            "server_epoch": "epoch-1",
        },
        "final_masked_emulator_dto": final_dto,
        "elapsed_s": 1.0,
    }


class CombatValueTrainingDataTest(unittest.TestCase):
    def test_loads_numeric_targets_skips_no_target_and_preserves_context(self) -> None:
        record = _record(
            _sample("a", 1, target_value=12.0, target_source="terminal"),
            _sample("b", 2, target_value=None, target_source="no_target"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            examples, stats = load_combat_value_examples([path])

        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(example.instance_id, "inst-1")
        self.assertEqual(example.decision_index, 0)
        self.assertEqual(example.root_decision_point_id, "d0")
        self.assertEqual(example.decision_point_id, "after-a")
        self.assertEqual(example.action["action_type"], "card")
        self.assertEqual(example.deepest_combat_depth, 3)
        self.assertEqual(example.dto_version, _DTO_VERSION)
        self.assertEqual(stats.root_samples, 2)
        self.assertEqual(stats.usable_samples, 1)
        self.assertEqual(stats.no_target_samples, 1)
        self.assertEqual(stats.dto_version, _DTO_VERSION)
        self.assertEqual(stats.upgraded_card_instances, 2)
        self.assertEqual(stats.enchanted_card_instances, 4)

    def test_terminal_post_state_without_next_decision_id_is_accepted(self) -> None:
        sample = _sample("a", 1, target_value=100.0, target_source="terminal")
        sample["decision_point_id"] = None
        sample["masked_emulator_dto"] = _dto(terminal=True)
        record = _record(sample)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "terminal.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            examples, _stats = load_combat_value_examples([path])
        self.assertIsNone(examples[0].decision_point_id)

    def test_nonterminal_post_state_without_next_decision_id_is_rejected(self) -> None:
        sample = _sample("a", 1, target_value=5.0, target_source="value_bootstrap")
        sample["decision_point_id"] = None
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "missing-decision.jsonl"
            path.write_text(json.dumps(_record(sample)) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-terminal post-state"):
                load_combat_value_examples([path])

    def test_no_target_with_numeric_value_fails_closed(self) -> None:
        record = _record(_sample("a", 1, target_value=0.0, target_source="no_target"))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "null target_value"):
                load_combat_value_examples([path])

    def test_mixed_dto_generations_are_rejected(self) -> None:
        first = _record(_sample("a", 1, target_value=1.0, target_source="value_bootstrap"))
        second = _record(
            _sample("b", 2, target_value=2.0, target_source="value_bootstrap"),
            dto_version="emulator-other",
        )
        sample = second["root_value_samples"][0]
        sample["masked_emulator_dto"] = dto_replace(
            sample["masked_emulator_dto"], dto_version="emulator-other"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            a = Path(tmpdir) / "a.jsonl"
            b = Path(tmpdir) / "b.jsonl"
            a.write_text(json.dumps(first) + "\n", encoding="utf-8")
            b.write_text(json.dumps(second) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mixes incompatible"):
                load_combat_value_examples([a, b])

    def test_sample_mask_is_checked_independently(self) -> None:
        sample = _sample("a", 1, target_value=1.0, target_source="value_bootstrap")
        sample["masked_emulator_dto"] = dto_replace(
            sample["masked_emulator_dto"], mask_version="1.1"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad-sample.jsonl"
            path.write_text(json.dumps(_record(sample)) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mask_version='1.2'"):
                load_combat_value_examples([path])

    def test_rl_loader_uses_actual_transition_not_counterfactual_root_sample(self) -> None:
        counterfactual = _sample("oracle-other", 1, target_value=50.0, target_source="terminal")
        record = _record(counterfactual)
        final_dto = _dto(terminal=True)
        record["runtime_transition"]["next_masked_emulator_dto"] = final_dto
        record["runtime_transition"]["combat_result"] = "victory"
        episode = _episode(
            final_dto=final_dto,
            decisions_collected=1,
            completed=True,
            combat_result="victory",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle.jsonl"
            path.write_text(json.dumps(record) + "\n" + json.dumps(episode) + "\n", encoding="utf-8")
            episodes = load_combat_value_rl_episodes([path])

        self.assertEqual(len(episodes), 1)
        self.assertTrue(episodes[0].usable_for_terminal_return)
        self.assertEqual(episodes[0].combat_result, "victory")
        self.assertEqual(len(episodes[0].steps), 1)
        self.assertEqual(episodes[0].steps[0].chosen_action_id, "actual")
        self.assertNotEqual(episodes[0].steps[0].chosen_action_id, "oracle-other")

    def test_rl_loader_keeps_truncated_episode_but_marks_not_terminal_return_usable(self) -> None:
        record = _record()
        episode = _episode(
            final_dto=record["runtime_transition"]["next_masked_emulator_dto"],
            decisions_collected=1,
            completed=False,
            combat_result=None,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle.jsonl"
            path.write_text(json.dumps(record) + "\n" + json.dumps(episode) + "\n", encoding="utf-8")
            episodes = load_combat_value_rl_episodes([path])
            completed = load_combat_value_rl_episodes([path], completed_only=True)
        self.assertEqual(len(episodes), 1)
        self.assertFalse(episodes[0].usable_for_terminal_return)
        self.assertEqual(completed, [])


if __name__ == "__main__":
    unittest.main()
