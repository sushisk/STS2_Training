# Stable Frontier Pruner

## 0. 文章の目的

この文書は `decision/stable_pruner.py`、`learned_pruner.py`、`pruner_features.py`、`pruner_training_data.py`、`pruner_rl.py` の実装 contract を説明する。対象は stable/resolved frontier の survivor 選択だけであり、Beam Search 全体は [02_decision_core.md](02_decision_core.md) を参照する。

## 1. 概要

`StableFrontierPruner` は、`BeamSearchEngine` が作った ordered stable frontier から残す node index を選ぶ public seam である。入力は immutable な `StablePruneNodeView` と `StablePruneContext` のみで、`BeamNode`、DTO payload、branch id、rng id、action payload、Whole Run capacity state は渡されない。

baseline は `ValueTopKPruner` で、`state_score` 降順に top K を返す。learned runtime は `LinearStableFrontierPruner` で、artifact schema v2 と feature schema v2 を検証してから dependency-free に `frontier_score` を計算する。RL exploration は `PlackettLuceLinearStableFrontierPruner` が同じ linear score から ordered sample without replacement を行う。

## 2. Architecture

`StablePruneNodeView` schema は `STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION = 1` である。フィールドは `value`、`root_action_id`、`depth`、`combat_depth`、`continuation_steps`、`terminal`、`action_type`、`policy_rank`、`policy_score`、`post_coverage_rank`、`candidate_source`。canonical property として `state_score`、`action_rank`、`action_score` を持つ。

`StablePruneContext` は `search_id`、`prune_step_id`、`phase`、`beam_width`、`max_depth`、`depths_completed`、`remaining_time_ms` を保持する。runtime の通常呼び出しでは `k == context.beam_width` である。

`pruner_features.py` の `PRUNER_FEATURE_SCHEMA_VERSION = 2` は 30 個の feature を固定する。内容は node state score、frontier-relative value stats/rank、root action group stats、depth/terminal、policy rank/score missingness、post coverage rank、candidate source、coarse action type、beam width、frontier size である。

`pruner_training_data.py` は Oracle JSONL から `PrunerFrontierTrainingExample` と pairwise examples を作る。`no_target` は pairwise label に使わず、terminal/value_bootstrap には configurable weight を付ける。`pruner_rl.py` は behavior artifact SHA と selection log probability を含む `PrunerRLStep` を記録する。

## 3. API

```python
class StableFrontierPruner:
    name = "stable_frontier_pruner"
    version = "1"
    def select(self, frontier: Sequence[StablePruneNodeView], *, k: int, context: StablePruneContext) -> list[int]

class ValueTopKPruner(StableFrontierPruner):
    name = "value_top_k"
    version = "1"
```

```python
stable_pruner_feature_matrix(
    frontier: Sequence[StablePruneNodeView],
    *,
    context: StablePruneContext,
) -> list[tuple[float, ...]]
```

```python
class LinearStableFrontierPruner(StableFrontierPruner):
    @classmethod
    def from_weights_file(cls, path: str | Path) -> "LinearStableFrontierPruner"
    def frontier_scores(self, frontier, *, context) -> list[float]
    def select(self, frontier, *, k: int, context) -> list[int]
```

```python
class PlackettLuceLinearStableFrontierPruner(StableFrontierPruner):
    @classmethod
    def from_weights_file(cls, path, *, temperature=1.0, seed=0, collector=None)
    def select(self, frontier, *, k: int, context) -> list[int]

plackett_luce_log_probability(scores, sampled_indices, *, temperature) -> float
plackett_luce_logprob_gradient(scores, feature_rows, sampled_indices, *, temperature, scale) -> tuple[float, ...]
```

## 4. 使用例

```python
from sts2_training.decision import (
    LinearStableFrontierPruner,
    StablePruneContext,
    StablePruneNodeView,
)

frontier = [
    StablePruneNodeView(3.0, "a", 1, 1, 0, False, "card", 0, 1.2, 0, "policy"),
    StablePruneNodeView(2.5, "b", 1, 1, 0, False, "system", 1, 0.4, 1, "policy"),
]
context = StablePruneContext(
    search_id="s1",
    prune_step_id="p1",
    phase="stable",
    beam_width=1,
    max_depth=3,
    depths_completed=1,
    remaining_time_ms=None,
)

pruner = LinearStableFrontierPruner.from_weights_file(
    "tools/output/stable_pruner_weights.json"
)
survivor_indices = pruner.select(frontier, k=1, context=context)
```

## 5. 補足説明

古い docs には Oracle v3/v4 という説明が残っているが、現在の `oracle_log.py` は `ORACLE_RECORD_SCHEMA_VERSION = 6` である。learned pruner 側の current contract は artifact schema `2`、feature schema `2`、node-view schema `1`。実行 CLI と学習 workflow は [07_runner_cli.md](07_runner_cli.md)、Oracle target の意味は [04_oracle.md](04_oracle.md) を参照する。
