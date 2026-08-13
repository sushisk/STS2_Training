from __future__ import annotations

import unittest

from sts2_training.decision.action_score_features import ACTION_SCORE_FEATURE_NAMES
from sts2_training.decision.action_score_training_data import (
    CombatActionScoreDatasetStats,
    CombatActionScoreTrainingExample,
)
from train_combat_action_score import evaluate_action_score_model, fit_action_score_model


def _example(action_id: str, estimated_q: float, *, action_card: float) -> CombatActionScoreTrainingExample:
    features = [0.0] * len(ACTION_SCORE_FEATURE_NAMES)
    features[ACTION_SCORE_FEATURE_NAMES.index("action_card")] = action_card
    return CombatActionScoreTrainingExample(
        source_path="oracle.jsonl",
        instance_id="instance-1",
        decision_index=0,
        decision_point_id="dp-1",
        action_id=action_id,
        action={"action_id": action_id, "action_type": "card" if action_card else "system"},
        features=tuple(features),
        estimated_q=estimated_q,
        target_source="terminal",
        sample_weight=1.0,
        censored=False,
        censor_reason=None,
        terminal_reached=True,
        dto_version="emulator-test",
    )


class TrainCombatActionScoreTest(unittest.TestCase):
    def test_pairwise_fit_learns_within_decision_order_and_reports_metrics(self) -> None:
        examples = [
            _example("best", 10.0, action_card=1.0),
            _example("worse", 1.0, action_card=0.0),
        ]
        fitted = fit_action_score_model(examples, c=1.0, tie_tolerance=1e-9)

        self.assertGreater(fitted.score(examples[0].features), fitted.score(examples[1].features))
        metrics = evaluate_action_score_model(
            fitted,
            examples,
            CombatActionScoreDatasetStats(
                decision_records=1,
                root_actions=2,
                usable_actions=2,
                no_target_actions=0,
                unresolved_actions=0,
                censored_actions=0,
                dto_version="emulator-test",
            ),
            tie_tolerance=1e-9,
        )
        self.assertEqual(metrics["pair_count"], 1)
        self.assertEqual(metrics["pairwise_accuracy"], 1.0)
        self.assertEqual(metrics["top1_accuracy"], 1.0)
        self.assertEqual(metrics["mean_top1_q_regret"], 0.0)


if __name__ == "__main__":
    unittest.main()
