from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from sts2_training.decision.action_score_features import (
    ACTION_SCORE_FEATURE_NAMES,
    ACTION_SCORE_FEATURE_SCHEMA_VERSION,
)
from sts2_training.decision.learned_policy import (
    ACTION_SCORE_MODEL_TYPE,
    LEARNED_ACTION_SCORE_ARTIFACT_SCHEMA_VERSION,
    LinearActionScorePolicy,
)
from sts2_training.decision.oracle_log import ORACLE_VALUE_MASK_VERSION

_DTO_VERSION = "emulator-test"


def _dto() -> dict:
    return {
        "mask_version": ORACLE_VALUE_MASK_VERSION,
        "dto_version": _DTO_VERSION,
        "hp": 40,
        "maxHp": 80,
        "block": 0,
        "energy": 3,
        "enemies": [],
        "hand": [],
        "drawPile": [],
        "discardPile": [],
        "exhaustPile": [],
        "potions": [],
        "playerPowers": [],
    }


def _payload() -> dict:
    coefficients = [0.0] * len(ACTION_SCORE_FEATURE_NAMES)
    coefficients[ACTION_SCORE_FEATURE_NAMES.index("action_card")] = 2.0
    return {
        "model_type": ACTION_SCORE_MODEL_TYPE,
        "artifact_schema_version": LEARNED_ACTION_SCORE_ARTIFACT_SCHEMA_VERSION,
        "feature_schema_version": ACTION_SCORE_FEATURE_SCHEMA_VERSION,
        "required_mask_version": ORACLE_VALUE_MASK_VERSION,
        "required_dto_version": _DTO_VERSION,
        "feature_names": list(ACTION_SCORE_FEATURE_NAMES),
        "coefficients": coefficients,
        "intercept": 0.0,
        "mean": [0.0] * len(ACTION_SCORE_FEATURE_NAMES),
        "scale": [1.0] * len(ACTION_SCORE_FEATURE_NAMES),
    }


def _model_from_payload(payload: dict) -> LinearActionScorePolicy:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "weights.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return LinearActionScorePolicy.from_weights_file(path)


class LinearActionScorePolicyTest(unittest.TestCase):
    def _model(self) -> tuple[LinearActionScorePolicy, bytes]:
        payload = _payload()
        raw = json.dumps(payload).encode()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.json"
            path.write_bytes(raw)
            model = LinearActionScorePolicy.from_weights_file(path)
        return model, raw

    def test_ranks_candidates_with_dependency_free_linear_score(self) -> None:
        model, raw = self._model()
        actions = [
            {"action_id": "end", "action_type": "system", "parameters": {}},
            {"action_id": "card", "action_type": "card", "parameters": {}},
        ]
        candidates = model.propose(actions, _dto(), top_k=2)
        self.assertEqual([candidate.action_id for candidate in candidates], ["card", "end"])
        self.assertEqual([candidate.action_score for candidate in candidates], [2.0, 0.0])
        provenance = model.oracle_provenance()
        self.assertEqual(provenance["artifact_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(provenance["required_dto_version"], _DTO_VERSION)

    def test_board_context_interaction_can_reverse_candidate_order(self) -> None:
        payload = _payload()
        payload["coefficients"] = [0.0] * len(ACTION_SCORE_FEATURE_NAMES)
        payload["coefficients"][ACTION_SCORE_FEATURE_NAMES.index("action_card")] = -0.5
        payload["coefficients"][
            ACTION_SCORE_FEATURE_NAMES.index("context_danger_ratio_x_action_card")
        ] = 1.0
        model = _model_from_payload(payload)
        actions = [
            {"action_id": "end", "action_type": "system", "parameters": {}},
            {"action_id": "card", "action_type": "card", "parameters": {}},
        ]

        low_danger = _dto()
        high_danger = _dto()
        high_danger["enemies"] = [
            {
                "index": 0,
                "hp": 20,
                "maxHp": 20,
                "block": 0,
                "isAlive": True,
                "intent": {"attackDamage": 40, "attackRepeats": 1},
                "powers": [],
            }
        ]

        self.assertEqual(
            [candidate.action_id for candidate in model.propose(actions, low_danger, top_k=2)],
            ["end", "card"],
        )
        self.assertEqual(
            [candidate.action_id for candidate in model.propose(actions, high_danger, top_k=2)],
            ["card", "end"],
        )

    def test_equal_scores_preserve_legal_action_order(self) -> None:
        payload = _payload()
        payload["coefficients"] = [0.0] * len(ACTION_SCORE_FEATURE_NAMES)
        model = _model_from_payload(payload)
        actions = [
            {"action_id": "b", "action_type": "system", "parameters": {}},
            {"action_id": "a", "action_type": "system", "parameters": {}},
        ]
        candidates = model.propose(actions, _dto(), top_k=2)
        self.assertEqual([candidate.action_id for candidate in candidates], ["b", "a"])
        self.assertEqual([candidate.action_score for candidate in candidates], [0.0, 0.0])

    def test_runtime_rejects_other_dto_generation(self) -> None:
        model, _ = self._model()
        with self.assertRaisesRegex(ValueError, "dto_version"):
            model.propose([], {**_dto(), "dto_version": "other"}, top_k=1)

    def test_feature_schema_mismatch_fails_closed(self) -> None:
        payload = _payload()
        payload["feature_schema_version"] = 999
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "feature schema mismatch"):
                LinearActionScorePolicy.from_weights_file(path)


if __name__ == "__main__":
    unittest.main()
