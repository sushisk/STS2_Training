import update_stable_pruner_rl
from sts2_training.decision.pruner_features import PRUNER_FEATURE_NAMES

def test_reinforce_update_moves_toward_rewarded_selection():
    width = len(PRUNER_FEATURE_NAMES)
    first = [0.0] * width; first[0] = 1.0
    step = {"frontier_features": [first, [0.0] * width], "sampled_indices": [0], "temperature": 1.0}
    example = update_stable_pruner_rl.RLUpdateExample(1.0, (step,), "x", 1)
    updated, result = update_stable_pruner_rl.apply_reinforce_update(
        [0.0] * width, [1.0] * width, [example], learning_rate=0.1, gradient_clip_norm=5.0)
    assert updated[0] > 0.0
    assert result.positive_reward_examples == 1
