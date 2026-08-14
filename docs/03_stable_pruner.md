# Stable Frontier Pruner

## 0. 文章の目的

この文書は `decision/stable_pruner.py`、`learned_pruner.py`、`pruner_features.py`、`pruner_training_data.py`、`pruner_rl.py` と runner 側の stable-pruner RL trajectory / reward を説明する。Beam Search 全体は [02_decision_core.md](02_decision_core.md)、実行入口は [07_runner_cli.md](07_runner_cli.md) を参照する。

## 1. 概要

`StableFrontierPruner` は ordered stable frontier から survivor index を選ぶ public seam である。baseline は `ValueTopKPruner`、learned runtime は `LinearStableFrontierPruner`、RL exploration は `PlackettLuceLinearStableFrontierPruner` を使う。

learned pruner の current contract は artifact schema `2`、feature schema `2`、node-view schema `1`。resource-aware RL trajectory は schema `3` で、paired baseline/learned outcome に terminal HP と potion retention を加えた reward を保存する。

## 2. Architecture

`StablePruneNodeView` は `value`、`root_action_id`、depth、terminal、action type、policy rank/score、post-coverage rank、candidate source などを持つ immutable view である。`StablePruneContext` は search/prune step、beam width、depth budget、remaining time を持つ。`pruner_features.py` はこれらから 30 個の schema-v2 feature を作る。

`pruner_training_data.py` は Oracle JSONL から supervised pairwise examples を作る。`pruner_rl.py` は behavior artifact SHA、frontier score、sampled survivor order、selection log probability などを記録する。Oracle target/provenance は [04_oracle.md](04_oracle.md) を参照する。

resource evaluator は version `1` で、terminal snapshot を `CombatResourceSnapshot(hp, max_hp, potion_count, initial_potion_count)` として固定する。`potions` は slot array のため `null` を empty slot として許可し、non-null mapping の数だけを current potion count とする。

```text
hp_fraction = clamp(hp / max_hp, 0, 1)
potion_fraction = 1                                           if initial_potion_count == 0
                  clamp(potion_count / initial_potion_count)  otherwise
resource_quality = 0.8 * hp_fraction + 0.2 * potion_fraction

reward = outcome_delta
       + 0.25 * resource_quality_delta
       - node_cost_weight * nodes_expanded_delta
       - beam_ms_cost_weight * beam_ms_delta
```

outcome は victory/win=`1.0`、defeat/loss=`0.0`。どちらかが不明なら reward は作らない。resource quality は `[0, 1]` なので resource term は `[-0.25, 0.25]` に収まり、terminal win/loss の outcome delta `±1` を resource term 単独で反転しない。

`ResourceCapturingABRunner` は paired A/B の terminal DTO を arm ごとに capture し、scenario 開始時の occupied potion 数を共通 denominator にして terminal resource fields を付与する。capture state は runner instance が所有する。

RL updater は behavior artifact/schema、node-view schema、sampled index order、temperature/sampler seed、selection log probability、paired result/reward decomposition、resource evaluator constants を検証し、意味の違う trajectory を fail closed で拒否する。

## 3. API

```python
class StableFrontierPruner:
    def select(self, frontier, *, k: int, context: StablePruneContext) -> list[int]

class LinearStableFrontierPruner(StableFrontierPruner):
    @classmethod
    def from_weights_file(cls, path: str | Path) -> "LinearStableFrontierPruner"
    def frontier_scores(self, frontier, *, context) -> list[float]

class PlackettLuceLinearStableFrontierPruner(StableFrontierPruner):
    @classmethod
    def from_weights_file(cls, path, *, temperature=1.0, seed=0, collector=None)
```

```python
@dataclass(frozen=True)
class CombatResourceSnapshot:
    hp: float
    max_hp: float
    potion_count: int
    initial_potion_count: int

combat_resource_snapshot(dto, *, initial_potion_count: int) -> CombatResourceSnapshot
combat_resource_quality(snapshot: CombatResourceSnapshot) -> float
paired_pruner_reward(
    pair,
    *,
    node_cost_weight: float = 0.0,
    beam_ms_cost_weight: float = 0.0,
) -> PairedPrunerReward | None
```

```python
plackett_luce_log_probability(scores, sampled_indices, *, temperature) -> float
plackett_luce_logprob_gradient(scores, feature_rows, sampled_indices, *, temperature, scale) -> tuple[float, ...]
```

## 4. 使用例

```python
from sts2_training.runner.combat_resource_reward import (
    combat_resource_quality,
    combat_resource_snapshot,
)

snapshot = combat_resource_snapshot(
    {
        "hp": 60,
        "maxHp": 80,
        "potions": [{"slot": 0, "potion_id": "A"}, None, None],
    },
    initial_potion_count=2,
)
quality = combat_resource_quality(snapshot)
```

stable-pruner の学習/RL/A-B 実行例は [07_runner_cli.md](07_runner_cli.md) を参照する。

## 5. 補足説明

current contract は learned-pruner artifact schema `2`、feature schema `2`、node-view schema `1`、RL trajectory schema `3`、resource evaluator version `1`。trajectory v3 は terminal resource fields と evaluator metadata を含むため、v2 trajectory を v3 として読み替えず再 collection する。

resource reward の係数は evaluator contract の一部である。意味を変える場合は evaluator/trajectory の versioning と updater validation を同時に更新する。Beam/Value/Policy の score semantics は [02_decision_core.md](02_decision_core.md)、Oracle supervised target は [04_oracle.md](04_oracle.md) を参照する。
