import json
import pytest
import update_stable_pruner_rl
from rl_v3_record_fixture import record
from sts2_training.decision.pruner_features import PRUNER_FEATURE_NAMES

def test_invalid_sampled_indices_are_rejected(tmp_path):
    value = record()
    value["steps"][0]["sampled_indices"] = [0, 0]
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sampled_indices must be unique valid frontier indices"):
        update_stable_pruner_rl.load_update_examples(
            [path], expected_artifact_sha256="expected",
            coefficients=[0.0] * len(PRUNER_FEATURE_NAMES),
            scale=[1.0] * len(PRUNER_FEATURE_NAMES),
        )
