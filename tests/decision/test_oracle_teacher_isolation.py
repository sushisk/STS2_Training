from __future__ import annotations

import json
import random
import unittest

from sts2_training.decision.beam_search import BeamSearchConfig, BeamSearchEngine
from sts2_training.decision.oracle_search import (
    BudgetedOracleCollector,
    OracleCollectionConfig,
    _oracle_provenance,
)
from sts2_training.decision.policy import ActionCandidate, PolicyModel, PriorHeuristicPolicy
from sts2_training.decision.value import HeuristicValueFunction, ValueModel


class _SimplePolicy(PolicyModel):
    def propose(self, legal_actions, masked_emulator_dto, *, top_k):
        return [
            ActionCandidate(action_id=action["action_id"])
            for action in legal_actions[:top_k]
        ]


class _StatefulValue(ValueModel):
    def __init__(self) -> None:
        self.calls = 0

    def evaluate_batch(self, dtos):
        self.calls += 1
        return [float(self.calls) for _dto in dtos]

    def oracle_provenance(self):
        return {"kind": "stateful-test"}


class _BadMetadataValue(ValueModel):
    def evaluate_batch(self, dtos):
        return [0.0 for _dto in dtos]

    def oracle_provenance(self):
        return {"bad": object()}


class OracleTeacherIsolationTest(unittest.TestCase):
    def test_from_beam_engine_copies_value_model_instead_of_sharing_runtime_state(self) -> None:
        runtime_value = _StatefulValue()
        engine = BeamSearchEngine(
            object(),
            policy=_SimplePolicy(),
            value_fn=runtime_value,
            config=BeamSearchConfig(beam_width=2, top_k_actions=2, max_depth=2),
        )

        oracle = BudgetedOracleCollector.from_beam_engine(
            engine,
            config=OracleCollectionConfig(
                beam_config=BeamSearchConfig(beam_width=2, top_k_actions=2, max_depth=2),
                target_beam_width=2,
            ),
        )

        self.assertIsNot(oracle._value_fn, runtime_value)  # noqa: SLF001
        oracle._value_fn.evaluate_batch([{}])  # noqa: SLF001
        self.assertEqual(runtime_value.calls, 0)

    def test_value_configuration_is_part_of_teacher_provenance(self) -> None:
        policy = PriorHeuristicPolicy(random.Random(123))
        first = HeuristicValueFunction(
            weights={"victory_bonus": 123.0},
            power_values={"POWER_A": 2.5},
        )
        second = HeuristicValueFunction(
            weights={"victory_bonus": 456.0},
            power_values={"POWER_A": 2.5},
        )

        first_provenance = _oracle_provenance(policy, first)
        second_provenance = _oracle_provenance(policy, second)

        self.assertEqual(
            first_provenance.teacher_value_metadata["weights"]["victory_bonus"],
            123.0,
        )
        self.assertEqual(
            first_provenance.teacher_value_metadata["power_values"]["POWER_A"],
            2.5,
        )
        self.assertNotEqual(
            first_provenance.teacher_value_metadata,
            second_provenance.teacher_value_metadata,
        )
        self.assertEqual(
            first_provenance.teacher_inner_policy_metadata["tie_break_rng"],
            "python_random",
        )
        self.assertIn(
            "rng_state_sha256",
            first_provenance.teacher_inner_policy_metadata,
        )
        json.dumps(first_provenance.teacher_value_metadata, allow_nan=False)

    def test_non_json_teacher_metadata_fails_closed(self) -> None:
        with self.assertRaisesRegex(TypeError, "JSON-serializable"):
            _oracle_provenance(_SimplePolicy(), _BadMetadataValue())


if __name__ == "__main__":
    unittest.main()
