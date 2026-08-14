from __future__ import annotations

import unittest
from pathlib import Path

from sts2_training.decision.learned_value import LEARNED_VALUE_ARTIFACT_SCHEMA_VERSION
from sts2_training.decision.oracle_log import ORACLE_VALUE_MASK_VERSION
from sts2_training.decision.value_features import VALUE_FEATURE_NAMES, VALUE_FEATURE_SCHEMA_VERSION
from sts2_training.decision.value_training_data import (
    CombatValueDatasetStats,
    CombatValueTrainingExample,
)
from train_combat_value import (
    SourceSplit,
    derive_terminal_values,
    evaluate_value_model,
    fit_value_model,
    split_source_files,
    weights_payload,
)


_DTO_VERSION = "emulator-test"


def _example(value: float, feature: float, *, source: str = "value_bootstrap"):
    return CombatValueTrainingExample(
        source_path="x.jsonl",
        instance_id="inst-1",
        decision_index=0,
        root_decision_point_id="root-d",
        decision_point_id="d",
        action_id=str(feature),
        action={"action_id": str(feature), "action_type": "card"},
        rng_id=int(feature) + 1,
        root_state_node_id=f"n-{feature}",
        features=(feature,) + (0.0,) * (len(VALUE_FEATURE_NAMES) - 1),
        target_value=value,
        target_source=source,
        sample_weight=1.0 if source == "terminal" else 0.5,
        terminal_reached=source == "terminal",
        deepest_combat_depth=2,
        censored=source != "terminal",
        censor_reason=None if source == "terminal" else "max_depth",
        best_node_id=f"leaf-{feature}",
        masked_emulator_dto={
            "mask_version": ORACLE_VALUE_MASK_VERSION,
            "dto_version": _DTO_VERSION,
        },
        dto_version=_DTO_VERSION,
    )


class TrainCombatValueTest(unittest.TestCase):
    def test_ridge_fit_and_metrics(self) -> None:
        examples = [_example(1.0, 0.0), _example(3.0, 1.0), _example(5.0, 2.0)]
        fitted = fit_value_model(examples, alpha=1e-6)
        metrics = evaluate_value_model(
            fitted,
            examples,
            CombatValueDatasetStats(1, 3, 3, 0, dto_version=_DTO_VERSION),
        )
        self.assertLess(metrics["rmse"], 1e-5)
        self.assertEqual(metrics["label_coverage"], 1.0)
        self.assertEqual(metrics["dto_version"], _DTO_VERSION)

    def test_artifact_records_feature_mask_and_dto_contract_versions(self) -> None:
        examples = [_example(1.0, 0.0), _example(3.0, 1.0)]
        fitted = fit_value_model(examples, alpha=1e-6)
        payload = weights_payload(
            fitted,
            metrics={"train": {"dto_version": _DTO_VERSION}},
            split=SourceSplit(train=[], val=[], test=[]),
            dataset_files=[],
            alpha=1e-6,
            terminal_weight=1.0,
            bootstrap_weight=0.5,
            seed=0,
            val_fraction=0.1,
            test_fraction=0.1,
            training_teacher_provenance={},
            dataset_teacher_provenance={},
            terminal_values={},
        )

        self.assertEqual(payload["artifact_schema_version"], LEARNED_VALUE_ARTIFACT_SCHEMA_VERSION)
        self.assertEqual(payload["artifact_schema_version"], 3)
        self.assertEqual(payload["feature_schema_version"], VALUE_FEATURE_SCHEMA_VERSION)
        self.assertEqual(payload["feature_schema_version"], 2)
        self.assertEqual(payload["required_mask_version"], ORACLE_VALUE_MASK_VERSION)
        self.assertEqual(payload["required_dto_version"], _DTO_VERSION)
        self.assertEqual(payload["metrics"]["train"]["dto_version"], _DTO_VERSION)

    def test_artifact_rejects_mixed_dto_generations_across_metrics(self) -> None:
        fitted = fit_value_model([_example(1.0, 0.0), _example(3.0, 1.0)], alpha=1e-6)
        with self.assertRaisesRegex(ValueError, "mix dto_version"):
            weights_payload(
                fitted,
                metrics={
                    "train": {"dto_version": _DTO_VERSION},
                    "val": {"dto_version": "emulator-other"},
                },
                split=SourceSplit(train=[], val=[], test=[]),
                dataset_files=[],
                alpha=1e-6,
                terminal_weight=1.0,
                bootstrap_weight=0.5,
                seed=0,
                val_fraction=0.1,
                test_fraction=0.1,
                training_teacher_provenance={},
                dataset_teacher_provenance={},
                terminal_values={},
            )

    def test_terminal_values_require_consistency(self) -> None:
        base = _example(100.0, 0.0, source="terminal")
        victory = CombatValueTrainingExample(
            **{
                **base.__dict__,
                "masked_emulator_dto": {
                    "mask_version": ORACLE_VALUE_MASK_VERSION,
                    "dto_version": _DTO_VERSION,
                    "terminal": True,
                    "outcome": "victory",
                },
            }
        )
        self.assertEqual(derive_terminal_values([victory]), {"victory": 100.0})

        inconsistent = CombatValueTrainingExample(
            **{
                **victory.__dict__,
                "action_id": "other",
                "action": {"action_id": "other"},
                "rng_id": 99,
                "target_value": 101.0,
            }
        )
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            derive_terminal_values([victory, inconsistent])

    def test_terminal_values_can_come_from_teacher_provenance(self) -> None:
        provenance = {
            "teachers": [
                {
                    "provenance": {
                        "teacher_value_metadata": {
                            "weights": {
                                "victory_bonus": 100000.0,
                                "defeat_penalty": -100000.0,
                            }
                        }
                    }
                }
            ]
        }
        self.assertEqual(
            derive_terminal_values([], provenance),
            {"victory": 100000.0, "defeat": -100000.0},
        )

    def test_split_is_source_file_based(self) -> None:
        paths = [Path(f"{index}.jsonl") for index in range(10)]
        split = split_source_files(paths, val_fraction=0.2, test_fraction=0.2, seed=0)
        self.assertEqual(len(split.train), 6)
        self.assertEqual(len(split.val), 2)
        self.assertEqual(len(split.test), 2)


if __name__ == "__main__":
    unittest.main()
