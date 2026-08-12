# Slay the Spire 2 デッキ評価・勝率推定モデル 実装案

## 0. 文章の目的

本書は、Slay the Spire 2 のRun中の状態から最終勝率を推定し、将来的にはカード報酬選択などの意思決定へ利用するための実装方針を整理する。

主な入力は以下を想定する。

- 現在Floor / Act
- HP / Max HP
- Gold
- デッキ
- レリック
- 次の数戦・Elite・Boss情報
- その他Run状態

基本出力は次の勝率とする。

```text
P(win | current state)
```

カード報酬選択まで扱う場合は、各行動について `Q(state, action)` を推定し、Card A / Card B / Card C / Skip の期待勝率を比較する。

最初の実装では、いきなり完全な強化学習にせず、RunのWin/Loseを教師ラベルとした **Value Functionの教師あり学習** から開始する。

---

## 1. 概要

### 1.a 全体構成

推奨構成は以下。

```text
Card Set
   ↓
Card Encoder
   ↓
Deck Encoder
   ↓
Deck Embedding ───────┐
                      │
Deck Summary ─────────┤
Relic Features ───────┤
Encounter Features ───┤
Floor / HP / Gold ────┤
                      ↓
                  Value MLP
                      ↓
                Win Probability
```

デッキは平均値や合計値だけには圧縮しない。**全カードの個別情報 + 人間が設計したデッキ集計特徴** のHybrid方式を基本とする。

### 1.b 学習データ

Runの最終結果を教師ラベルにする。

```text
Win  = 1
Lose = 0
```

同じRun内の各Floor状態に最終結果を対応させる。ただし同一Run内のFloorは強く相関するため、Train / Validation / Testは必ず **Run単位** で分割する。

例:

```text
Train      80%
Validation 10%
Test       10%
```

Floor単位でランダム分割すると、同じデッキの近い状態がTrainとValidationへ混ざり、評価が過大になる可能性がある。

### 1.c 評価指標

Accuracyだけではなく、確率予測として評価する。

- Log Loss
- Brier Score
- Calibration
- ROC-AUC

特に重要なのはCalibration。例えば70%と予測した状態群が実際にも約70%勝っているかを見る。

---

## 2. デッキとカードの入力設計

### 2.a カード1枚の表現

カードごとに固定長の特徴ベクトルを作る。

例:

```text
card_id
upgrade
cost

damage
block
draw
energy_gain

is_attack
is_skill
is_power

is_aoe
is_multi_hit
exhaust
retain
innate

strength_scaling
dexterity_scaling
poison
discard
exhaust_generation
その他効果
```

Card IDは整数値をそのまま数値特徴として渡さず、Embeddingにする。

```python
nn.Embedding(num_cards, 32)
```

概念:

```text
Card ID Embedding
+
Mechanical Features
       ↓
Shared Card MLP
       ↓
Card Vector
```

### 2.b 参照型カード

以下のようなカードを固定damage/block値だけに変換しない。

```text
デッキ内Attack枚数 × 3 Damage
Combat中Exhaustした枚数 × 2 Block
このTurnに使用したAttack数に応じて効果増加
```

カード定義を、**基本効果 + 参照対象 + 係数 + 条件 + Scope** として保持する。

例:

```yaml
effect_type: DAMAGE
base_value: 0

scaling:
  source: DECK_COUNT
  filter:
    card_type: ATTACK
  coefficient: 3
```

Queryの例:

```text
DECK_COUNT(type=ATTACK)
DECK_COUNT(cost=0)
COMBAT_COUNT(event=EXHAUST)
COMBAT_COUNT(event=ATTACK_PLAYED)
TURN_COUNT(event=CARD_PLAYED)
PLAYER_VALUE(STRENGTH)
PLAYER_VALUE(MISSING_HP)
```

Scopeは例えば以下。

```text
RUN_STATIC
FLOOR
COMBAT
TURN
```

Floor時点で計算可能な値は、事前計算した値も補助入力として与える。一方、Combat依存の未来値は0にせず、`is_dynamic`、`reference_scope`、`reference_type`、`coefficient` として構造を保持する。

### 2.c デッキ集計特徴

カードSetとは別に集計値も入力する。

例:

```text
deck_size
attack_count / skill_count / power_count
attack_ratio / skill_ratio / power_ratio
cost_0_count / cost_1_count / cost_2_count / cost_3plus_count
total_base_damage
damage_per_energy
total_block
block_per_energy
draw_amount
energy_generation
aoe_count
multi_hit_count
exhaust_enable
exhaust_payoff
discard_enable
discard_payoff
```

平均だけではなく、**合計・密度・最大値・枚数・コスト効率・分布** を使う。

---

## 3. ニューラルネットワーク実装

### 3.a Shared Card MLP

全カードに同じMLPを適用する。これをShared Card MLPとする。

```python
import torch
import torch.nn as nn


class CardEncoder(nn.Module):
    def __init__(self, num_cards, feature_dim):
        super().__init__()

        self.id_embedding = nn.Embedding(
            num_embeddings=num_cards,
            embedding_dim=32
        )

        self.mlp = nn.Sequential(
            nn.Linear(32 + feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, card_ids, card_features):
        id_vec = self.id_embedding(card_ids)
        x = torch.cat([id_vec, card_features], dim=-1)
        return self.mlp(x)
```

概念的には、すべてのカードが同じネットワークを共有する。

```text
Card 1 → Shared MLP → 64次元
Card 2 → Shared MLP → 64次元
...
Card N → Shared MLP → 64次元
```

### 3.b Deck Encoder

デッキ枚数はRunごとに異なる。

```text
20枚 → [20, 64]
30枚 → [30, 64]
45枚 → [45, 64]
```

そこでPoolingを使い、固定次元に変換する。

```python
sum_vec = card_vectors.sum(dim=0)
mean_vec = card_vectors.mean(dim=0)
max_vec = card_vectors.max(dim=0).values
```

各ベクトルが64次元なら、SUM / MEAN / MAXを結合して192次元にする。

```python
deck_vector = torch.cat([
    sum_vec,
    mean_vec,
    max_vec
])
```

役割は概ね以下。

```text
SUM  → デッキ全体に能力がどれだけ存在するか
MEAN → 能力の密度
MAX  → 1枚でも突出した能力を持つカードが存在するか
```

MINは初期実装では不要。

### 3.c 勝率予測MLP

Deck Vectorと他の状態情報を結合する。

```text
deck_vector
deck_summary
relic_features
encounter_features
floor
HP
gold
その他
```

これをValue MLPへ渡す。

```python
self.value_mlp = nn.Sequential(
    nn.Linear(input_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)
```

学習時は `nn.BCEWithLogitsLoss()` を使用し、推論時は `torch.sigmoid(logit)` で0〜1の勝率へ変換する。

---

## 4. 学習手順と発展方針

### 4.a 初期実装

最初は以下の構成をBaselineとする。

```text
Card Mechanical Features
+
Card ID Embedding
       ↓
Shared Card MLP
       ↓
64-d Card Vector
       ↓
SUM / MEAN / MAX
       ↓
192-d Deck Vector

        +

Engineered Deck Features
Relic Features
Floor / HP / Gold
Future Encounter Features

        ↓

Value MLP

        ↓

Win Probability
```

実装順:

1. カード効果データ形式を決める
2. 静的カード特徴を作る
3. Deck Summaryを作る
4. Card Encoderを実装
5. Deck Encoderを実装
6. Win/Lose教師あり学習を動かす
7. Run単位Validationを構築
8. Calibrationを確認する

### 4.b データ増加時の発展

Engineered FeaturesからCard ID方式へ完全に切り替えず、Hybridを維持する。

独立Run数の大まかな目安:

| 独立Run数 / キャラ | 方針 |
|---:|---|
| ～5,000 | Engineered Features中心 |
| 5,000～20,000 | 小さいCard Embedding追加 |
| 20,000～50,000 | Hybrid本格運用 |
| 50,000～100,000 | ID側の自由度を増やす |
| 100,000+ | カード間Interactionを強化 |
| 数十万+ | Set Transformer等を検討 |

実際の移行判断はRun数だけでなく、ValidationでHybridが安定して改善するかで判断する。

### 4.c 将来の強化学習化

カード報酬選択まで扱う場合、`state + action` から期待勝率を予測する。

```text
Action 1 = Card A
Action 2 = Card B
Action 3 = Card C
Action 4 = Skip
```

各候補について `Q(state, action)` を計算して最大のものを選ぶ。

さらにデータが増えれば、Set Transformer等を用いてカード間相互作用を直接学習させる。

```text
Strength生成 × Multi-hit
Exhaust generator × Exhaust payoff
```

またCombat依存参照については、将来的に別のCombat Predictorで `expected_exhaust_count`、`expected_cards_played`、`expected_combat_length` などを推定し、参照型カードの期待効果へ利用できる。
