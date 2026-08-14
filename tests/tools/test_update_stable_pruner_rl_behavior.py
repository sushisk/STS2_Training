import json
import pytest
import update_stable_pruner_rl
from sts2_training.decision.pruner_features import PRUNER_FEATURE_NAMES
from rl_v3_record_fixture import record

def test_behavior_artifact_mismatch_is_rejected(tmp_path):
    value = record(sha="other")
    path = tmp_path / "r.jsonl"; path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match update base"):
        update_stable_pruner_rl.load_update_examples([path], expected_artifact_sha256="expected", coefficients=[0.0] * len(PRUNER_FEATURE_NAMES), scale=[1.0] * len(PRUNER_FEATURE_NAMES))
