import json
import pytest
import update_stable_pruner_rl
from sts2_training.decision.pruner_features import PRUNER_FEATURE_NAMES
from rl_v3_record_fixture import record

def load_one(tmp_path, value):
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return update_stable_pruner_rl.load_update_examples([path], expected_artifact_sha256="expected", coefficients=[0.0] * len(PRUNER_FEATURE_NAMES), scale=[1.0] * len(PRUNER_FEATURE_NAMES))

def test_v3_reward_is_accepted(tmp_path):
    assert load_one(tmp_path, record())[0].reward == pytest.approx(0.2)
