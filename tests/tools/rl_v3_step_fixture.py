from sts2_training.decision.learned_pruner import LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION
from sts2_training.decision.pruner_features import PRUNER_FEATURE_NAMES, PRUNER_FEATURE_SCHEMA_VERSION
from sts2_training.decision.stable_pruner import STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION

def behavior(sha="expected"):
    return {"artifact_sha256": sha, "artifact_schema_version": LEARNED_PRUNER_ARTIFACT_SCHEMA_VERSION,
        "stable_prune_node_view_schema_version": STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
        "feature_schema_version": PRUNER_FEATURE_SCHEMA_VERSION, "temperature": 1.0, "sampler_seed": 17}

def step():
    rows = [[0.0] * len(PRUNER_FEATURE_NAMES), [0.0] * len(PRUNER_FEATURE_NAMES)]
    return {"beam_width": 1, "stable_prune_node_view_schema_version": STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION,
        "feature_schema_version": PRUNER_FEATURE_SCHEMA_VERSION, "temperature": 1.0, "sampler_seed": 17,
        "frontier_features": rows, "behavior_scores": [0.0, 0.0], "sampled_indices": [0],
        "returned_indices": [0], "selection_log_probability": -0.6931471805599453}
