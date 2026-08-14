import json
import pytest
import update_stable_pruner_rl
from rl_v3_record_fixture import record
from sts2_training.decision.pruner_features import PRUNER_FEATURE_NAMES
from sts2_training.decision.stable_pruner import STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION

def test_node_view_schema_mismatch_is_rejected(tmp_path):
    value = record()
    value["behavior"]["stable_prune_node_view_schema_version"] = STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION + 1
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stable_prune_node_view_schema_version mismatch"):
        update_stable_pruner_rl.load_update_examples(
            [path], expected_artifact_sha256="expected",
            coefficients=[0.0] * len(PRUNER_FEATURE_NAMES),
            scale=[1.0] * len(PRUNER_FEATURE_NAMES),
        )
