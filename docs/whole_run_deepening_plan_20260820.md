# Whole Run 到達階層 深化計画（P1）

P0（Beam scope 修正）後の `data/evaluation/score_logs/p0-verify`（5戦 / 453 探索 / 7,392 trace イベント）
を解析し、次に到達階層を押し上げるための計画を示す。

## 0. 出発点

| 条件 | 平均到達階層 | 内訳 |
|---|---:|---|
| P0 修正前（`top_k=8`, 5戦） | 5.2 | 4, 8, 6, 3, 5 |
| P0 修正後（`top_k=8`, 1戦） | 9.0 | 9 |
| P0 修正後（`top_k=4`, 5戦） | **7.0** | 11, 5, 5, 8, 6 |

いずれも全戦敗北。死亡階層は 5, 5, 6, 8, 11 で、Act 1 ボス（16F 前後）に届いていない。

scope 系の健全性は回復している。

- `branches_faulted` = 0
- `branches_out_of_scope` = 4（全て後述 §1.7 の map_select 遷移。戦闘中の攻撃 drop はゼロ）
- root 選択の最頻手が `STRIKE_IRONCLAD`（91回）になり、`SLIMED` を最善手に選ぶ挙動は消滅した

## 1. p0-verify ログの解析結果

### 1.1 最大の戦術的損失: 打てるカードを手札に残したままターン終了

`End Turn` が best root action になった **85 回すべて**で、支払い可能なカードが手札に残っていた。

| 残されたカード | 件数 |
|---|---:|
| `STRIKE_IRONCLAD` | 92 |
| `DEFEND_IRONCLAD` | 89 |
| その他（`POKE`, `BATTLE_TRANCE`, `SHRUG_IT_OFF`, `ANGER`, `SECOND_WIND` ほか） | 約 60 |

`is_available` は Emulator の `CardModel.CanPlay()`（`HasEnoughResourcesFor` を含む、
`CardModel.cs:1787`）の結果であり、`BuildLegalActions` も `IsAvailable` で filter している
（`GameInstance.cs:5441`）。つまり**エネルギーは足りていた**。

Slay the Spire では、打てる Strike / Defend を残してターンを渡すのはほぼ常に損である。
これが現在の最大の漏れ。

### 1.2 原因: leaf の評価時点がそろっていない

深さ1ノードの value 平均:

| 種別 | 件数 | 平均 | 中央値 |
|---|---:|---:|---:|
| カード | 623 | **+0.40** | +3.36 |
| `End Turn` | 309 | **−14.72** | −10.50 |

深さ1では カード枝が `End Turn` より平均 **+18.1**（中央値 +19.3）有利であるにもかかわらず、
最終 leaf 比較では `End Turn` が勝つ（85回中 53回はカード枝が pruning を生き延びた上で負けている）。

さらに、

- **25%（108/439）の探索で `best_value` が深さ1の最良ノード値を下回る**（平均 −11.8）
- 深さ1 greedy と beam の選択が **51%（224/439）一致しない**

つまり現状は「深く探索するほど評価が悪化する」状態であり、探索が価値関数を修正するのではなく、
価値関数の欠陥を探索が増幅している。

**メカニズム（解析的に確認できる）**

エネルギー 1、手札に `DEFEND`（block 5）、敵の攻撃予告 `I=12`、現在 HP `H`、maxHp 80 とする。
現行の重み（`player_hp_ratio` 40.0 / `player_block` +0.5 / `predicted_incoming_damage` −1.0、
`incoming_damage = max(0, I − block)`）で計算すると:

| 手順 | 状態 | value |
|---|---|---:|
| A-1: `DEFEND` を打つ | hp=H, block=5, I=12 | `0.5H + 2.5 − 7` = **0.5H − 4.5** |
| A-2: A-1 → `End Turn` | hp=H−7, block=0, I'=12 | `0.5H−3.5 − 12` = **0.5H − 15.5** |
| B-1: 先に `End Turn` | hp=H−12, block=0, I'=12 | `0.5H−6 − 12` = **0.5H − 18.0** |
| B-2: B-1 → `DEFEND` | hp=H−12, block=5, I'=12 | `0.5H−6 + 2.5 − 7` = **0.5H − 10.5** |

深さ2の leaf は A-2 = 0.5H−15.5、B-2 = 0.5H−10.5。**先にターンを終えた方が勝つ。**

現実には A も B も「カードを 1 枚打ち、敵の攻撃を 1 回受けた」状態で等価のはずである。
差が出るのは、`End Turn` 側の leaf だけが「敵の攻撃を受けた後、さらに次ターンの手番を 1 回得た状態」
になっているから。**偶数深さでは常に `End Turn` 側が半手先行して評価される。**

これは重みのチューニングでは直らない。leaf の意味を揃える必要がある。

### 1.3 block の二重計上

`incoming_damage = max(0, 攻撃予告 − block)` が重み −1.0 で入るのに加え、`player_block` が +0.5 で
独立に入っている（`decision/value.py:28`, `decision/combat_observation.py:172`）。

maxHp 80 では `player_hp_ratio` 40.0 → **1 HP = 0.5 点**。したがって、

- 実際に受け止める block 1 点 = 0.5 + 1.0 = **1.5 点 = HP 3 点分**
- 余る block 1 点 = 0.5 点 = HP 1 点分

深さ1 greedy が `DEFEND` を 139/439 で選ぶ（beam は 55）のはこの過大評価が原因。

### 1.4 敗因は「2手先の判断ミス」ではなく消耗

- 5戦すべてで、**最後の探索の `best_value` が −100,000**（全枝が敗北）
- それ以前に `best_value` が敗北になったのは 1〜3 回のみ
- prune 対象ノード 3,615 件のうち、敗北 leaf を含む prune イベントは **0 件**

2手先で回避できる死に方はしていない。数ターンかけて HP を失い、詰んだ局面で初めて敗北が見える。
**多ターンにわたる HP 管理が効いていない**ことの直接的な証拠であり、これも §1.2 の帰結である
（block と HP の交換レートが turn-invariant でないため、探索が生存を最適化できない）。

### 1.5 学習済み policy も同じ偏りを継承している

`action_score` ログ 368 件:

- policy 1位ラベル: `STRIKE_IRONCLAD` 143 / **`End Turn` 102** / `BASH` 36 / `DEFEND_IRONCLAD` 29
- **カードが打てる局面 346 件のうち、policy が `End Turn` を 1 位に置くのが 80 件（23%）**
- `End Turn` の policy 順位は平均 3.44

learned `action_score` は Oracle の `estimated_q` の順位を distill したものであり、その Oracle Q は
同じ `HeuristicValueFunction` と（P0 修正前の）beam で作られている。
**価値関数を直しても、artifact を作り直さない限り policy 側の `End Turn` 偏重は残る。**

### 1.6 戦闘外の意思決定に heuristic が無い action type がある

Emulator が発行する action type と `HeuristicCombatSelector` の対応:

| action type | 扱い |
|---|---|
| `card` / `choice_card` / `choice_confirm` / `choice_skip` | 専用 heuristic あり |
| `map_room` | `room_preference_scores` |
| `choice_event_option` | lethal 回避後に一様ランダム |
| `choice_reward_card` / `_potion_take` / `_potion_replace` / `_skip` | 専用 policy あり |
| **`choice_rest_option`** | **一様ランダム** |
| **`choice_shop_buy_card` / `_buy_relic` / `_buy_potion` / `_remove_card` / `choice_shop_leave`** | **一様ランダム** |
| **`choice_event_throw_potion`** | **一様ランダム** |

`HeuristicCombatSelector.select()` の分岐に該当がなく、末尾の `_choose(actions)`（一様ランダム）に落ちる。
**焚き火の「休息 vs 強化」とショップの購入・カード削除がコイン投げで決まっている。**
到達階層が深くなるほど効いてくる。

加えて、`floor_reach_eval` は `HeuristicCombatSelector(random.Random(seed))` を既定の
`epsilon=0.1` で構築しているため（`heuristic_selector.py:64`）、評価中も 10% は一様ランダムになる。

### 1.7 out-of-scope 4件について

4件は同一探索（run 00003, `d-root-000176`）で、`choice_card` 候補
（`RAMPAGE` / `BATTLE_TRANCE` / `TRUE_GRIT` / `PERFECTED_STRIKE`）を解決した先が `map_select` になったもの。
探索は `reason=not_beam_searchable` で終わり、`best_root_action_id=None` → `heuristic_fallback` に落ちている。

「Combat Beam の責務を保つなら非戦闘では beam を使わない」という方針に同意する。
ただし**現状すでに実害はない**（正しく fallback している）。真の問題は落ちた先の heuristic が
デッキ文脈を見ていないこと（§1.6）なので、優先度は低い。scope に `map_room` を足すのは反対で、
`ValueModel` の意味領域は戦闘状態であり、map 状態に heuristic value を当てても意味がない。

### 1.8 学習済み board score は現状使えない（artifact 実測）

`tools/output/combat_value_weights_oracle_v7_20260819.json`（`ridge_linear_combat_value`）を検査した。

| split | usable_samples | r² | MAE |
|---|---:|---:|---:|
| train | 5,581 | **0.175** | 29,753 |
| val | 604 | 0.241 | 30,293 |
| test | 716 | 0.215 | 24,405 |

標準化係数（絶対値上位と、戦闘評価上重要なもの）:

| feature | 係数 |
|---|---:|
| `energy_present` | **+16,859** |
| `lethal_threat` | −15,994 |
| `enemy_hp_ratio` | −15,292 |
| `enemies_alive` | −11,026 |
| **`player_hp_ratio`** | **−7,448**（符号が逆） |
| **`player_block`** | **−1,531**（符号が逆） |
| `incoming_damage` | −491 |

HP と block という戦闘評価の中核 2 feature の符号が反転し、最大係数がフィールド有無フラグ
`energy_present` になっている。board score として成立していない。

**原因: ラベルのスケール混在**

Oracle `root_value_samples` 6,901 件（`no_target` 1,117 件を除く）の内訳:

| `target_source` | 件数 | 平均 | 中央値 | 範囲 |
|---|---:|---:|---:|---|
| `terminal` | 1,308 (19%) | +58,104 | 100,000 | ±100,000 |
| `value_bootstrap` | 5,593 (81%) | +10.7 | 14.0 | −76.5 〜 +106.5（sd 23.6） |

桁が約 4,000 倍違う 2 population を 1 本の ridge 回帰（二乗誤差）に入れている。
損失への寄与比はおよそ `0.19 × 100,000²` : `0.81 × 23.6²` ≈ **4×10⁶ : 1**。
`bootstrap_weight=0.5` はこれをさらに悪化させる方向に効く。
非終端の順序関係は損失にほぼ寄与せず、ノイズにフィットしている。

- intercept が +15,299 ≒ ラベル全体平均（`0.19×58,104 + 0.81×10.7` ≈ 11,048）
- 最大係数が `energy_present` = 実質「終端か否か」の判別器
- r² 0.17〜0.24 は、その終端判別を部分的に説明しているだけ

**決定的な点**: `LinearValueModel.evaluate` は終端 DTO を `exact_terminal_utility` で短絡する
（`decision/learned_value.py:99`）。損失の 99.99% を占める 19% の終端ラベルは、
**推論時に一度も使われない**。モデルが実際に働く 81% の非終端が、損失にほぼ寄与していない。

**天井と床**

censor 内訳を見ると `value_bootstrap` 5,593 件はすべて `value_bootstrap:max_depth` であり、
深さ 4 の leaf を `HeuristicValueFunction` で評価した値そのものである。したがって:

- **天井** = ラベルが heuristic の出力 → 学習済み value は原理的に heuristic を超えられない。
  §1.2 のターン境界の半手ずれと §1.3 の block 二重計上もラベルに焼き込まれている。
- **床** = ラベルのスケール混在 → 現状は heuristic の再現すらできていない（HP・block の符号反転）。

「学習が遅かった」のではなく、**現行の目的関数では収束先が board score にならない**。
対処は §P1-B。

## 2. 計画

### P1-A 価値関数を turn-invariant にする（最優先）

**A1. `effective_hp` 単一項化** — **実装済み（2026-08-20）。実装記録は §6。**

`player_hp_ratio` / `player_block` / `predicted_incoming_damage` の 3 項を、

```
effective_hp = hp − max(0, 敵の攻撃予告合計 − block)
effective_hp_ratio = effective_hp / maxHp        # 重み 40.0
```

の 1 項に置き換える。§1.2 の例で検算すると、

| 手順 | effective_hp |
|---|---|
| A-1: `DEFEND` | H − max(0, 12−5) = **H − 7** |
| A-2: A-1 → `End Turn` | (H−7) − 12 = **H − 19** |
| B-1: `End Turn` | (H−12) − 12 = **H − 24** |
| B-2: B-1 → `DEFEND` | (H−12) − 7 = **H − 19** |

深さ2の leaf が A-2 = B-2 = H−19 で**一致**し、深さ1では A-1 (H−7) > B-1 (H−24) と
正しく「先に防御する方が安全」になる。ターン境界の半手ずれと block の二重計上が同時に消える。

変更は `decision/value.py` の 1 ファイル。`CombatObservation.incoming_damage` は既に
`max(0, incoming_before_block − block)` なので、`effective_hp = hp − incoming_damage` で表せる。

推奨する重み:

```python
DEFAULT_WEIGHTS = {
    "effective_hp_ratio": 40.0,   # maxHp 80 で 1 HP = 0.5 点
    "enemy_hp_ratio": -30.0,
    "enemies_alive": -2.0,
    "buff_debuff_score": 2.0,
    "enemy_buff_debuff_score": 2.0,
    "named_power_score": 1.0,
    "victory_bonus": 100_000.0,
    "defeat_penalty": -100_000.0,
}
```

**A2. 受け入れ基準（score ログで直接検証できる）**

- `End Turn` が best root になった局面で「支払い可能なカードが残っている」件数: 85 → 大幅減
- `best_value < 深さ1 最良値` の割合: 25% → 大幅減
- 深さ1ノードの平均 value の カード vs `End Turn` 差が縮む

いずれも 1〜2 戦で確認できる。到達階層より先にこちらを見るべき。

**A3. leaf のターン整列（A1 で不十分な場合）**

A1 は静的近似なので、複数ターン先の状態（block の持ち越し、次の intent の変化）までは揃わない。
残る場合は、mid-turn leaf に対して `End Turn` を強制的に 1 手延長してから評価する
（quiescence 相当）。leaf ごとに emulate が 1 回増えるためコストは大きく、A1 の効果を見てから判断する。

### P1-B 学習成果物の作り直しと value 学習目的関数の変更（A の後、必須）

**結論: 学習済み board score（`--board-score learned`）は現時点では使用しない。**
`--board-score heuristic` の運用を維持する。理由は §1.8 のとおり、天井（ラベルが heuristic 出力）と
床（ラベルのスケール混在）の 2 つが同時に壊れているため。

なお、Oracle collect は `instance_type="combat"` の scenario instance で走っており
（`runner/scenario.py:156`）、`_is_whole_run_unresolved_out_of_scope` は `whole_run` でしか発火しない。
さらに `runner/episode.py:build_engine` 経由なので scope も広い。
**Oracle データセット自体は P0 の scope 欠陥の影響を受けていない。**
作り直しが必要な理由は value 関数とラベル設計であって、scope ではない。

**B1. value 学習から終端ラベルを外す（最優先・単独で実施可能）**

`LinearValueModel.evaluate` は終端 DTO を `exact_terminal_utility` で短絡する
（`decision/learned_value.py:99`）。つまり損失の 99.99% を占める終端ラベル（19%）は
**推論時に一度も使われない**。回帰から外して非終端 `value_bootstrap` のみで学習する。
推論時の終端値は従来どおり exact ±100,000 を返せばよいので、挙動は変わらない。

**B2. 目的関数を絶対値回帰から順位学習に変える**

beam search と stable pruner が使うのは「1 探索内の相対順序」だけであり、絶対値の回帰精度は
目的関数として誤っている。`action_score` 側は既にこの判断で pairwise logistic distillation を
採用している（`docs/02_decision_core.md` §1）が、value 側だけ絶対値 ridge 回帰のままだった。

- 学習: 同一 decision / 同一探索内の pairwise ranking
- 評価指標: MAE / RMSE ではなく Spearman 順位相関、pairwise accuracy

終端と非終端のスケール差（±100,000 vs ±100）は**推論時には正しい**（確定勝ちが常に勝つ）。
壊れるのは学習の損失だけなので、B1 で学習集合から外すだけで済む。

**B3. A1 適用後に Oracle を取り直す**

`value_bootstrap` ラベル 5,593 件はすべて `value_bootstrap:max_depth`、すなわち深さ 4 の leaf を
`HeuristicValueFunction` で評価した値そのもの。**完璧に学習しても heuristic を超えられない。**
A1（`effective_hp`）を入れてから収集し直すことで初めて天井が上がる。

1. A1 適用 → Oracle collect 再実行（`scripts/build_oracle_dataset.ps1` /
   `scripts/run_oracle_collection_managed.ps1`）
2. B1 + B2 の目的関数で value を再学習、`action_score` / stable pruner も再学習
3. `--board-score learned` と `heuristic` を同条件で再比較

**B4. feature schema v3 の検討**

value feature schema v2 は `player_hp_ratio` / `player_block` / `incoming_damage` を別 feature に持つ。
線形モデルでは `max(0, 攻撃予告 − block)` の非線形性を表現できないため、
`effective_hp_ratio` を feature として与える v3 を検討する。
`incoming_damage` は既に `max(0, ·)` 済み（`combat_observation.py:172`）なので、
`effective_hp_ratio = (hp − incoming_damage) / maxHp` を 1 feature 足すだけでよい。

### P1-C 戦闘外方策（Aと並行可能・独立）

1. **`choice_rest_option` の heuristic**（最優先）。最小構成として
   「HP 比が閾値未満なら休息、そうでなければ強化」。
2. **`choice_shop_*` の heuristic**。カード削除を最優先、次に relic、`skada_score` 上位のカード、
   所持金が足りなければ `choice_shop_leave`。
3. `choice_event_throw_potion` は当面「投げない」を既定にする。
4. **報酬カードの skip**。`CardDataRewardCardSelectionPolicy` は必ずカードを取る。
   「デッキ枚数が閾値超 かつ `skada_score` が閾値未満なら skip」を入れる。
5. **評価中の ε を 0 にする**（P0.5 未実施）。`floor_reach_eval` に `--eval-epsilon`（既定 0.0）。

1〜3 は現状「一様ランダム」なので、雑な heuristic でも期待値は必ず上がる。

### P1-D 探索予算

A1 適用後に `--beam-depth 1 / 2 / 3` を同条件で比較する。
現状は深さを足すほど leaf 評価が壊れているため（§1.2）、A1 の前に深さを増やしても意味がない。
逆に A1 後は深さ 3 の効果が正しく出るはず。

`evaluate_whole_run.py` に `--time-budget-ms` を追加して探索コストの上限を切れるようにする
（`BeamSearchConfig.time_budget_ms` は既にあるが CLI から指定できない）。

### P1-E 計測基盤

1. **`--detailed-log-dir` を有効にして評価する。** rootごとの完全な DTO と action score を同一ログに保存し、
   「どの戦闘でどれだけ HP を失ったか」「消耗死か事故死か」を後から判定できるようにする。
2. n=30 のベースラインを取り直す。現在の n=5 は母標準偏差が大きく（P0 後 5戦の分散 5.2）、
   平均 7.0 の差は判定に足りない。

## 3. 実施順序

```
P1-A1 (value.py)  →  1〜2戦で §A2 の受け入れ基準を検証
    → 通れば P1-C1/C4/C5（rest / reward skip / ε=0）を投入   ← ここまでで一度 n=30
    → P1-B3（A1 後に Oracle 再収集）
        → P1-B1 + B2（終端ラベル除外 + 順位学習）で value 再学習
        → action_score / stable pruner も再学習
        → learned / heuristic を同条件で再比較               ← n=30
    → P1-D（深さ ladder）                                    ← n=30
    → P1-C2/C3（shop / potion throw）、P1-A3・P1-B4（必要なら）
```

P1-A1 と P1-C は独立なので並行できるが、**測定は必ず分けて行う**。同時に入れると
どちらが効いたか分からなくなる。

**それまでの間は `--board-score heuristic` を使う。** §1.8 のとおり、現行の学習済み
board score は HP・block の符号が反転しており、heuristic より明確に劣る。
P1-B1 と B2（目的関数の変更）は Oracle 再収集を待たずにオフラインで実施・検証できるので、
A1 と並行して進めてよい（既存データで順位相関が改善するかは今の artifact でも測れる）。

## 4. 計算コスト

P0 後の実測: 1 探索あたり約 2.7〜3.3 秒、1 戦あたり探索 59〜165 回 → **1 戦 2.5〜9 分**。

| 条件 | 見積もり |
|---|---|
| n=5（受け入れ確認用） | 15〜45 分 |
| n=30（正式比較） | 1.5〜4.5 時間 |
| n=30 × 深さ 1/2/3 | 半日〜1 日 |

サーバは `GameInstance` 共有のため concurrency 1 固定。深さ ladder は
`--time-budget-ms` で上限を切るか、n=20 に落として回すのが現実的。
また `evaluate_whole_run.py` は全 run 終了後に一括で JSON を書くため、
途中で打ち切ると結果が全部失われる（`topk8-20260820` の事例）。
**run ごとの逐次書き出しを先に入れておくこと。**

## 5. 優先度まとめ

| # | 項目 | 根拠 | 見込み |
|---|---|---|---|
| 1 | `effective_hp` 化（A1） | §1.1〜1.4。打てる Strike 92 / Defend 89 を捨てている | 大 |
| 2 | rest / reward skip / ε=0（C1, C4, C5） | §1.6。焚き火がコイン投げ | 中〜大 |
| 3 | value 目的関数の変更（B1, B2） | §1.8。終端ラベルが損失の 99.99% を占め、推論では未使用 | 中〜大 |
| 4 | Oracle 再収集 + 再学習（B3） | §1.5 / §1.8。policy が End Turn を 23% で 1 位、ラベルの天井が heuristic | 中〜大 |
| 5 | 深さ ladder（D） | §1.2。A1 前は測っても無意味 | 中 |
| 6 | shop heuristic（C2） | §1.6。出現頻度は低い | 小〜中 |
| 7 | selection ログ + n=30（E） | 判断の前提 | 測定基盤 |
| 8 | feature schema v3（B4） | §1.8。線形では `max(0,·)` を表現できない | 小〜中 |
| 9 | 非戦闘での beam 停止（§1.7） | 実害なし | 小 |

**当面の運用**: `--board-score heuristic` を維持する（§1.8 / §P1-B）。

## 6. P1-A1 実装記録（2026-08-20）

### 6.1 変更内容

**`src/sts2_training/decision/value.py`**

`DEFAULT_WEIGHTS` から `player_hp_ratio` / `player_block` / `predicted_incoming_damage` を削除し、
`effective_hp_ratio: 40.0` を追加。`_extract_features()` は

```python
effective_hp = observation.hp - observation.incoming_damage
...
"effective_hp_ratio": effective_hp / observation.max_hp,
```

を返す。`CombatObservation.incoming_damage` が既に
`max(0, 敵の攻撃予告合計 − block)` なので、追加の計算は不要
（`combat_observation.py:172`）。

`enemy_hp_ratio` / `enemies_alive` / `buff_debuff_score` / `enemy_buff_debuff_score` /
`named_power_score` / `victory_bonus` / `defeat_penalty` は変更なし。

module docstring に、なぜ HP と block を別項にしてはいけないか（ターン偶奇で勝敗が決まる）、
旧構成での二重計上、0 で clamp しない理由を記載した。

**設計上の決定**

- **`effective_hp` を 0 で clamp しない。** 負値は「公開されている敵のターンが致死」を意味する。
  clamp すると確定死の局面で被弾を減らす勾配が消えるため、負のまま扱う。
- **予備の block 項を置かない。** Oracle 訓練データの実 DTO 483 件すべてで敵 intent が公開されて
  いることを確認した（うち攻撃 intent 373 件、非攻撃 intent 110 件）。したがって
  「intent が無いため block が無価値になる」縮退は起きない。非攻撃 intent に対する block が
  0 点になるのは仕様として正しい。

### 6.2 テスト

`tests/decision/test_value.py` に 5 件追加。

| テスト | 内容 |
|---|---|
| `test_survival_score_is_turn_invariant` | §1.2 の例（H=60, maxHp=80, intent 12, block 5）。深さ2の leaf が一致し、深さ1では先に防御した側が上 |
| `test_block_trades_one_for_one_against_hp_it_prevents` | block +5 と hp +5 が同値 |
| `test_block_beyond_incoming_damage_is_worthless` | block 12 → 30 で値が変わらない |
| `test_block_against_a_non_attacking_intent_is_worthless` | 非攻撃 intent に対する block 20 が 0 点 |
| `test_lethal_incoming_damage_keeps_a_gradient_below_zero` | hp 10 / intent 40 でも block 15 が改善になる |

`test_survival_score_is_turn_invariant` は旧重みでは失敗する
（旧: A-2 = 14.5 vs B-2 = 19.5 で「先にターンを終える」が勝つ）。

既存テストの追従:

- `test_value.py::test_block_is_consumed_once_across_multiple_enemy_attacks` を
  `effective_hp_ratio` 基準に書き換え（block 10 対 合計 20 被弾 → 0.8。攻撃者ごとに block を
  消費する実装なら 1.0 になる）
- `test_value.py::test_custom_weights_override_defaults` と
  `test_review_hardening.py::test_non_finite_weight_is_rejected` の weight 名を差し替え

全体: 779 passed / 6 skipped。

実 DTO 2,059 件（`data/.oracle_training_input_v7`）で例外ゼロを確認。
非終端の値域は [−152.75, +44.84]、平均 3.04、中央値 8.86。

### 6.3 ドキュメント

- `docs/02_decision_core.md` §5.2 に「生存項は turn-invariant でなければならない」を追加
- `src/sts2_training/decision/how_to_use.md` の既定 Value の説明を更新
- `decision/value.py` の module docstring に理由と旧構成の問題を記載

### 6.4 影響範囲の注意

`HeuristicValueFunction.oracle_provenance()` は weights を返すため、**Oracle teacher provenance が
変わる**。これは意図した変更で、新旧の Oracle レコードが provenance で区別できる。

既存 artifact（`combat_value_weights_oracle_v7_20260819.json` など）は
旧 heuristic のラベルで学習されているため、**そのまま使い続けない**。§P1-B の順序に従う。

### 6.5 次の検証

```
python tools/evaluate_whole_run.py --character-id IRONCLAD --num-runs 2 \
    --models learned --search-modes standard --board-score heuristic \
    --beam-depth 2 --beam-width 8 --top-k-actions 4 \
    --output-dir data/evaluation/whole_run/a1-verify \
    --detailed-log-dir data/evaluation/detailed_logs/a1-verify
```

§A2 の受け入れ基準を score ログで確認する。

1. `End Turn` が best root になった局面で支払い可能なカードが残っている件数（p0-verify: 85/85）
2. `best_value < 深さ1 最良値` の割合（p0-verify: 25%、平均 −11.8）
3. 深さ1ノードの平均 value の カード（+0.40）vs `End Turn`（−14.72）の差

到達階層より先にこの 3 つを見る。1 と 2 が改善していなければ、
`effective_hp` だけでは不足で A3（leaf のターン整列）に進む判断材料になる。

## 7. A1 検証結果と P1-C 第一弾（2026-08-20, a1-verify-2）

### 7.1 scope の健全性

`tools/check_score_log_scope.py`: `out_of_scope_drops = 0`、`branch_faults = 0`。
（a1-verify（1回目）は commit 3492141 の `isinstance(search_mode, BeamSearchConfig)` ガードにより
scope が狭いまま実行されていたため無効。`evaluate_whole_run._mode_configs()` で widening を
適用し、`floor_reach_eval` に構築時 WARNING を追加して再発を検知できるようにした。）

### 7.2 A1（`effective_hp`）の効果

| 指標 | p0-verify (439探索) | a1-verify-2 (111探索) |
|---|---:|---:|
| 深さ1 value: カード | +0.40 | **+2.80** |
| 深さ1 value: `End Turn` | −14.72 | **−3.77** |
| 両者の差 | 15.12 | **6.57**（−57%） |
| `best_value < 深さ1最良値` | 25%、平均 −11.79 | 24%、平均 **−5.04**（−57%） |
| beam ≠ 深さ1 greedy | 51% | 43% |
| root 選択率 `STRIKE_IRONCLAD` | 20.5% | **33.3%** |
| root 選択率 `DEFEND_IRONCLAD` | 12.5% | 6.3% |
| root 選択率 `End Turn` | 19.4% | 18.0% |

**構造的な歪みは半減した。** ターン境界の非対称と block 二重計上が消え、攻撃の選択率が
1.6 倍になり、過大評価されていた `DEFEND` が半減した。

**残った半手ずれ**: `End Turn` の選択率はほぼ変わらず（19.4% → 18.0%）、
打てるカードを残したターン終了も 20 件残っている。深さ1で平均 +8.25 有利なカード枝が
depth-2 leaf 比較で負ける構図は縮小したが消えていない。
A1 は静的近似であり、block の持ち越しや次 intent の変化までは揃えられない。
これは §P1-A の A3（leaf のターン整列）が必要な残差である。

### 7.3 selection ログによる敗因の特定（新規）

`--detailed-log-dir` を有効にして HP 推移を初めて取得した。

| run | 階層ごとの HP |
|---|---|
| 0 | 80 → 57 → 43 → 40 → 9 → **6F で死亡** |
| 1 | 80 → 68 → 67 → 67 → 47 → **6F で死亡** |

- **HP は一度も回復していない。** maxHp も 80 のまま。
- 1 戦闘あたり 12〜42 HP を失い、6 階層で約 75 HP を消耗して死ぬ。
- 戦術的な事故死ではなく、純粋な消耗死（§1.4 の推定を裏付け）。

### 7.4 支配的な原因: map routing が戦闘を最大化していた

`selection/room_heuristic.py` の `_POINT_TYPE_SCORE` は `Unknown` を **−1.0**、
`Monster` を **0.0** としていた。つまり `?` より確定戦闘を好む。

Emulator の実装（`Core.Odds/UnknownMapPointOdds.cs:21-34`）では、`?` の抽選は

| 結果 | 確率 |
|---|---:|
| Event | 約 85% |
| Monster | 10% |
| Shop | 3% |
| Treasure | 2% |
| Elite | 発生しない |

`?` が戦闘になるのは 10 回に 1 回、`Monster` は毎回。期待 HP 損失は `?` の方が明確に小さく、
かつ `?` は道中で HP・レリック・金貨を得られる唯一の経路である。

2 戦の map 選択のうち、選択肢が分かれた 3 回すべてで `Monster` を選んでいた。
スコア表が決定的なので、確率ではなく**構造的に毎回そうなる**。

### 7.5 実装（P1-C 第一弾）

**1. `Unknown` を `Monster` より上に**（`selection/room_heuristic.py`）

```
Treasure 6.0 > RestSite 5.0 > Shop 3.0 > Unknown 2.0 > Monster 0.0 > Elite -2.0
```

`Unknown` は確定的な良部屋（Treasure/RestSite/Shop）より下、`Monster` より上。
module docstring に Emulator の抽選確率と、この順序を誤ると戦闘数が最大化される理由を記載。

a1-verify-2 のログで再生したところ、**選択肢が分かれた 3 回すべてが `Monster` → `Unknown` に反転**した。

**2. `--eval-epsilon`（既定 0.0）**（`floor_reach_eval` / `evaluate_whole_run`）

`HeuristicCombatSelector` の `epsilon=0.1` が評価中も有効で、map_room 選択とカード fallback の
10% が一様ランダムになっていた（§1.6）。評価は方策そのものを測るべきなので既定を 0.0 にし、
データ収集用途のために CLI から指定できるようにした。`run_floor_reach_eval` は範囲外の値を拒否する。

**テスト**: room heuristic 3件、epsilon 3件（+subtests）。全体 793 passed / 6 skipped。

### 7.6 次の候補（今回は未実装・証拠のみ）

測定を分離するため、以下は次のラウンドに回す。

**a. ショップ** — run 1 の 4F で `REMOVE_CARD` が選択肢にあるにもかかわらず、
`TREMBLE`（カード購入）を選び、次に `LEAVE_SHOP` を選んだ。
`choice_shop_*` は `HeuristicCombatSelector` に分岐がなく一様ランダム（§1.6）。
デッキ圧縮はカード購入より価値が高いので、`REMOVE_CARD` を最優先にする。

**b. 報酬カードの skip** — 6 階層で 5 枚追加（`BATTLE_TRANCE`, `HEADBUTT`×2, `BURNING_PACT`,
`CRUELTY`, `TREMBLE`×2, `BEGONE`, `AFTERLIFE`）。`Skip` は常に選択肢にあるが一度も選ばれていない。
デッキ希釈で初期 `STRIKE`/`DEFEND` の引きが悪くなる。

**c. A3（leaf のターン整列）** — §7.2 の残差。A1 だけでは `End Turn` の選択率が下がらない。

**d. 焚き火** — 6F までに `rest_choice` は一度も出現しなかった。
深く到達するようになってから効いてくるので、優先度は a/b より後。

## 8. route-verify 結果（2026-08-20, n=5）

### 8.1 到達階層

| 条件 | n | 平均 | 内訳 |
|---|---:|---:|---|
| p0-verify（P0 のみ） | 5 | 7.0 | 11, 5, 5, 8, 6 |
| a1-verify-2（+ effective_hp） | 2 | 6.0 | 6, 6 |
| **route-verify（+ ε=0 + routing）** | 5 | **13.2** | 13, 17, 8, 11, 17 |

中央値 13、最小 8、最大 17、標準偏差 3.49。全 5 戦とも Act 1（`act_index=0`）。
**2 戦が 17F（Act 1 ボス）まで到達**した。

p0-verify との差 +6.2 は Welch t = 2.98、df 6.9、**p = 0.021**。
ただし route-verify は effective_hp + ε=0 + routing の 3 変更を含む束であり、
単一要因の効果ではない。近因が routing であることは §8.2 のログで確認できる。

### 8.2 routing 変更の効果（selection ログ）

| | a1-verify-2（変更前） | route-verify（変更後） |
|---|---|---|
| 入った部屋 | Monster 9, Shop 1 | Monster 24, **Unknown 13**, **RestSite 11**, Treasure 4, Shop 4, Elite 3, Boss 2 |
| 選択肢が分かれた map 選択 | 3 → Monster 3 | 16 → **Unknown 7, RestSite 5**, Shop 2, Monster 2 |

**HP が回復するようになった。** 変更前は単調減少しかしていない。

| run | 階層ごとの HP |
|---|---|
| 0 | 80 → 60 → 64 → 64 → 50 → 50 → **70 → 70 → 80 → 80 → 80** → 32 → 13 |
| 1 | 80 → 56 → … → 29 → 29 → **53** → 17 → **41** → 20 → 8 → 8 → 8 |
| 4 | 80 → 68 → … → **69 → 78** → 67 → … → **72 → 92** → 92 → 66 → 66 → 4 |

run 4 は maxHp が 80 を超えて 92 に達している（`?` 由来のイベント/レリック）。
`?` と RestSite に寄る経路を選べるようになったことが、消耗死からの脱出そのものである。

（参考: 変更前は 80 → 57 → 43 → 40 → 9 → 2 で 6F 死亡。）

### 8.3 標準出力の WARNING について（無害）

route-verify で `out_of_scope_drop` が 44 件出たが、**全件が同一の無害クラス**である。

| 項目 | 値 |
|---|---|
| 落ちた先の boundary | `map_select` ×44 |
| その状態の action types | `['map_room']` ×44 |
| 落とした側の action_type | `choice_card` ×44 |
| `allowed_action_types` | 全 7 種（正しい） |
| `branch_faults` | 0 |

`allowed_action_types` が全 7 種なので **scope 設定は正常**。これは §1.7 と同じ事象で、
Neow やイベントの「デッキからカードを選ぶ」決定が `boundary=pending_choice` を持つため
Combat Beam が入口で受け入れてしまい、解決した先が `map_select` になって落ちる。
探索は `reason=not_beam_searchable` で終わり `heuristic_fallback` に落ちるので、**挙動は正しい**。

DTO を比較すると判別材料は明確である。

| | 戦闘中の `pending_choice` | 非戦闘の `choice_card` |
|---|---|---|
| `enemies` / `hand` / `drawPile` / `exhaustPile` | あり | **キー自体が無い** |
| `combatRoundNumber` / `turnNumber` | あり | **キー自体が無い** |
| `combatSessionId` | `None`（マスク済み） | `None` |

`combatSessionId` はマスクされていて使えないが、`combatRoundNumber`（または `enemies` キー）の
有無で戦闘中かどうかを判別できる。Whole Run の beam 受け入れ条件を
「boundary が stable/pending_choice」だけでなく「実際に Combat DTO であること」にすれば、
非戦闘のカード選択を beam が展開しなくなり、5 戦あたり 44 回の無駄な emulate とログノイズが消える。

**優先度は低い**（挙動は既に正しい）が、ログノイズが本物の scope 異常を隠すため、
scope 回帰を 2 回起こしている経緯からは片付けておく価値がある。

### 8.4 新たに判明した最優先項目: 焚き火が一様ランダム

routing 修正で RestSite に 11 回入れるようになった結果、`choice_rest_option` が
初めて実際に効くようになった。そして §1.6 のとおりこれは**一様ランダム**である。

| HP | 選択 |
|---:|---|
| 70 | HEAL |
| 80 | SMITH |
| 29 | HEAL |
| 17 | HEAL |
| **8** | **SMITH** |
| **18** | **SMITH** |
| **30** | **SMITH** |
| 67 | SMITH |
| 67 | SMITH |
| 72 | HEAL |
| 66 | SMITH |

合計 SMITH 7 / HEAL 4。**HP 8 で SMITH を選んでいる。**
低 HP（30 以下）の 5 回のうち 3 回が SMITH。
17F で死んだ 2 戦は HP 8 と 4 で ボスに入っており、これが直接効いている可能性が高い。

最小構成の対処: HP 比が閾値未満なら `HEAL`、そうでなければ `SMITH`。

### 8.5 その他の未処理項目（証拠更新）

- **報酬カードの skip: 5 戦 26 回の報酬選択で `Skip` を 0 回**しか選んでいない。デッキは 26 枚増加。
- **ショップ**: `LEAVE_SHOP` 4、ポーション/カード購入 7。依然として一様ランダムで、
  `REMOVE_CARD` を選んだ回数は 0。
- `heuristic_fallback` の決定数が 1 戦あたり 10〜25 に増加した（変更前は 3〜11）。
  深く到達するほど戦闘外の意思決定の比重が上がるため、8.4 / 8.5 の価値は今後さらに増す。

## 9. 焚き火ヒューリスティックと評価の並列化（2026-08-20）

### 9.1 焚き火（§8.4 の対処）

新規 `selection/rest_heuristic.py`。`choice_rest_option` に分岐が無く一様ランダムだった。

- 回復系（`HEAL` / `MEND`、いずれも `maxHp * 0.3` 回復）は**欠けている HP の割合**で採点
- `SMITH` は定数 3.0 → 交差点は **HP 70%**
- それ以外（`LIFT` / `DIG` / `CLONE` / `COOK` / `HATCH` / `KINDLE`）は 0.0。
  比較できる根拠が無いため推測しない（`room_heuristic` の未知 point_type と同じ扱い）
- HP が読めない場合は回復（致命的になり得ない側）

route-verify の実データ 11 件を再生した結果、**5 件が変化**した。

| HP | 変更前 | 変更後 |
|---:|---|---|
| 8 (10%) | SMITH | **HEAL** |
| 18 (22%) | SMITH | **HEAL** |
| 30 (33%) | SMITH | **HEAL** |
| 70 (88%) | HEAL | **SMITH** |
| 72 (78%) | HEAL | **SMITH** |

低 HP の 3 件（うち 1 件は HP 8）がすべて回復に、高 HP の無駄な回復 2 件が強化に変わった。

### 9.2 評価の並列化（`--ports`）

**計測**: 1 戦あたりの `emulate_actions` は 322〜662 リクエスト、1 リクエスト 0.2〜0.6 秒。
バッチサイズは平均 7.4、最大 32 で、これは `beam_width × top_k_actions` = 8×4 = 32 が上限。
`max_batch_size`（64）には届いていないので、**1 リクエストあたりの仕事量は既に飽和**している。

**制約**: RL サーバ 1 台は並列化できない。`API/tcp_server.py:213` の `_call_handler` が
`_handler_lock`（1 個の `asyncio.Lock`）で全接続の全リクエストを直列化しており、
Emulator は spawn された CLR プロセス 1 個の中で動く（`API/api_runtime.py`）。
Emulator 側も `RunManager.Instance` などの singleton を持つため、これは意図的な設計と考えられる。
したがって 1 ポートに `--concurrency` を上げてもキューが伸びるだけである。

**実装**: `--ports 8765,8766,8767,8768` を追加し、worker をポートに 1 対 1 で固定する。

- `--concurrency` の既定を「ポート数」に変更（ポート未指定なら 1）
- 重複ポートは拒否（2 worker が同じサーバを共有するのを防ぐ）
- `--concurrency` がポート数を超える場合は WARNING を出して続行
- `floor_reach_eval` と `evaluate_whole_run` の両方に追加。レポート JSON に `ports` を記録

サーバを N 個立てれば N 戦が同時に走る。1 戦 2.5〜9 分なので、n=30 は
4 プロセスで **1.5〜4.5 時間 → 25〜70 分**が見込める。

### 9.3 テスト

- `tests/selection/test_rest_heuristic.py` 9 件（交差点 70%、MEND、HP 不明時、未知オプション）
- `tests/selection/test_heuristic_selector.py` に 2 件（selector が新分岐を通ること、ε=0 で決定的）
- `tests/runner/test_floor_reach_eval_sharding.py` 7 件
  （worker とポートの 1 対 1、concurrency の既定、単一ポートは直列、
  超過時の WARNING、重複/不正ポートの拒否）

全体 811 passed / 6 skipped。

### 9.4 サーバ起動の内製化（`--start-rl-servers`）

サーバを N 個手で立てるのは評価のたびに面倒で、1 台立て忘れるとその worker の担当分が
丸ごと接続エラーになる。`tools/evaluate_whole_run.py` に起動・停止を持たせた。

```
python tools/evaluate_whole_run.py --character-id IRONCLAD --num-runs 10 \
    --start-rl-servers 4 --rl-root C:\STS2_RL ...
```

checkout の場所は `--rl-root` か `STS2_RL_ROOT` で渡す。シェルごとに環境変数の設定構文が
違う（PowerShell は `$env:STS2_RL_ROOT = "..."`、`set` は `Set-Variable` のエイリアスで
環境変数にはならない）ため、引数で渡すほうが確実。

新規 `runner/rl_server_pool.py`。`tests/integration/_paired_rl_helpers.py` の既存パターンを
踏襲しつつ、production 用に 3 点を明示的に扱う。

1. **サーバ出力はファイルへ。`PIPE` は使わない。** 評価中は誰も stdout を読まないため、
   OS バッファが埋まると子プロセスが次の write でブロックし、実行中の全リクエストが
   クライアント側タイムアウトまでハングする（サーバ側デッドロックにしか見えない）。
   ログは `--output-dir/rl-server-<port>.log` に残す。
2. **停止はプロセスツリー全体。** Whole Run サーバは `Run/worker_pool.py` の
   multiprocessing worker を持ち、その子が CLR ハンドルを握っている。親だけ terminate すると
   worker が孤児化してコアとメモリを占有し続ける。Windows は `taskkill /F /T`、
   POSIX は プロセスグループへのシグナルで落とす。
3. **ポートごとに起動完了を待つ。** CLR 初期化は遅く、同時起動でさらに遅くなるので
   spawn 完了 ≠ 受付可能。各ポートが accept するまでポーリングし、起動中に死んだ場合は
   タイムアウトではなくそのサーバのログ末尾を添えて報告する。

`--ports` を併用すると、そのポートで起動する（数が合わなければ拒否）。
`--rl-root` / `STS2_RL_ROOT` で checkout を指定する。

`--ports` のみ（サーバは自分で管理）の場合は、開始前に全ポートへ素の TCP 疎通確認を行い、
listen していないポートを名指しして即座に失敗する。セッションを消費しないよう
ハンドシェイクはしない。

**テスト**: `tests/runner/test_rl_server_pool.py` 16 件。実際に subprocess を起動する
stub checkout で、起動→listen 待ち→ツリー停止、起動時死亡時のログ添付、
listen しない場合のタイムアウト、例外時もサーバが残らないこと、
および `evaluate_whole_run` 側の配線を検証する。全体 830 passed / 6 skipped。

## 10. A3: leaf のターン整列（2026-08-20）

### 10.1 何を直すか

固定深さの探索は、行が違えばターン内の違う地点で止まる。`max_depth=2` では

| 経路 | 到達点 |
|---|---|
| カード → カード | まだ自ターン中。**敵の攻撃予告は未払い** |
| カード → `End Turn` | 攻撃を受けた後 |
| `End Turn` → カード | 攻撃を受けた後、**さらに次ターンの手番を 1 回得た状態** |

この 3 つを同じ尺度で比べると、root action は打ち筋の良し悪しではなく**ターンの偶奇**で決まる。
§7.2 で effective_hp が歪みを半減させたあとも `End Turn` の選択率が下がらず、
打てるカードを手札に残す挙動が 20 件残っていたのはこれが原因。

### 10.2 実装

`BeamSearchEngine._align_leaf_turns()`。遅れている leaf を、その行がどのみち指す
**強制 `End Turn` の先で採点し直す**。探索の追加ではなく quiescence であり、
node は深さも木の中の位置も保ったまま `state_score` だけが決着後の値に置き換わる。

- **持ち上げは最大 1 ターン**（`min(turn) + 1` まで）。それ以上延長すると
  「プレイヤーが意図的にパスするターン」を模擬することになり、
  予算をカードプレイに使った行こそが不当に罰される。目標ターン以上の leaf はそのまま
  （越えたターン境界は戻せない）。
- **ベストエフォート**。ブランチが fault したり batch が reject された場合は
  元のスコアを残す。候補手を落とすより、精密化を諦めるほうが安全。
- ターン判定は DTO の `turnNumber`（全 Combat DTO に存在し `combatRoundNumber` と一致）。
  `End Turn` は Emulator が Combat 決定に必ず 1 つだけ発行する `action_type="system"`
  （`GameInstance.BuildLegalActions`、実データ 7103/7103 が `End Turn`）。
- コストは 1 探索あたり **emulate バッチ 1 回**（1 探索は元々 7 回程度）。

`BeamSearchConfig.turn_aligned_leaves`（既定 `False`）と
`evaluate_whole_run.py --turn-aligned-leaves` で切り替える。A/B を成立させるため既定は off。
`BeamSearchStats.leaves_turn_aligned` と `search_end` trace に発火回数を記録する。

### 10.3 検証

**単体**: `tests/decision/test_turn_aligned_leaves.py` 6 件。
「カードを打つ行が有利に見えるのは攻撃が未払いだからで、払わせると答えが反転する」
という比較そのものを fake client で再現し、off なら `card`(30.0)、on なら `end`(20.0) を
選ぶことを assert する。全 leaf が同一ターンなら追加リクエストが出ないこと、
alignment batch が fault しても元のスコアが残ることも含む。

**実機**: 実 Emulator に対する 1 戦（41 探索）で

| 指標 | 値 |
|---|---:|
| alignment が発火した探索 | 32 / 41（78%） |
| 再採点された leaf 合計 | 233 |
| 1 探索あたり | 最大 9 |

発火しなかった 9 件は全 leaf が同一ターンだったケース。
`turnNumber` の読み取り、`End Turn` の検出、延長バッチの成功が実データで確認できた。

全体 836 passed / 6 skipped。

### 10.4 次の測定

on/off を同条件で比較する。`--turn-aligned-leaves` 以外の引数は揃えること。
n=10 では CI が ±2.8 あり、rest-verify（10.8）との差を見るには足りない可能性が高いので、
効果量が小さい場合は n=20〜30 が要る（4 並列で 50 分程度）。

score ログで先に見るべき受け入れ基準（rest-verify 比）:

1. `End Turn` が best root になり、かつ支払い可能なカードが手札に残っている件数
2. `best_value < 深さ1 最良値` の割合
3. 深さ1 の カード と `End Turn` の平均 value 差

### 10.5 測定結果: **A3 は効かなかった**（2026-08-20, a3-verify n=10）

| 指標（1 探索あたりに正規化） | A3 off (rest-verify, 942探索) | A3 on (a3-verify, 627探索) |
|---|---:|---:|
| **`End Turn` 選択かつ打てるカードが残る** | 145 (**15.4%**) | 106 (**16.9%**) |
| `End Turn` が best root | 220 (23.4%) | 165 (26.3%) |
| 深さ1 の カード − `End Turn` 平均 value 差 | 10.58 | 8.29 |
| branch faults | 0 / 964 | 3 / 644 |

**A3 が狙った指標（1 行目）は改善せず、わずかに悪化した。**

到達階層は 10.8 → 7.8（`[2,3,5,5,6,7,7,9,17,17]`）、Welch t = −1.45、**p = 0.167**。
有意ではないが方向は悪い。

`best_value < 深さ1最良値` が 23% → 45% に増えるのは**測定上の見かけ**である。
prune trace の深さ1ノードは整列されていない一方 leaf は整列後なので、
両者はもはや同じ尺度ではない。この数字は A3 on では有効な指標にならない。

3 件の fault（`Illegal action: 2` / `worker_exception`）はすべて 1 つの run に集中しており、
その run は floor 17 に到達している。10 戦とも最後の探索は `best_value = −100,000` で、
死因は従来どおり「詰んだ局面」であって異常終了ではない。fault は floor 低下の原因ではない。

### 10.6 なぜ効かなかったか

`turnNumber` による整列は**吸収した敵ターン数**を揃えるが、
**そのターン内で消費したプレイヤー行動数**を揃えない。`max_depth=2` では:

| 経路 | 整列後 |
|---|---|
| `[カード, カード]` | +`End Turn` → **ターン2の開始時**（ターン1でカード2枚） |
| `[カード, End Turn]` | 既にターン2 → **ターン2の開始時**（ターン1でカード1枚） |
| `[End Turn, カード]` | 既にターン2 → **ターン2の途中**、既にカード1枚を消化済み |

3 行目だけが「ターン2で 1 枚打った効果」を leaf に含んでいる。
**A3 が消そうとした半手ずれがそのまま残っている。** 整列できたのは
1 行目と 2 行目の軸だけで、そこは元々問題ではなかった。

固定の**行動**深さ + ターンごとに可変な行動数、という組み合わせがある限り、
`End Turn` の延長だけでは等化できない。

### 10.7 次に取るべき形（提案）

`End Turn` を深さ予算から外し、その上で turn 整列する。

- 現状 `End Turn` は `system` なので `combat_depth` を 1 消費する
  （continuation だけが消費しない扱い）。これを continuation と同様に「消費しない」に変える
- すると `[カード, カード]`（ターン1）と `[End Turn, カード, カード]`（ターン2）が
  どちらも**カード 2 枚消化**で並ぶ
- そこに A3 の turn 整列を掛けると、両者は「カード 2 枚消化・敵ターン 1 回吸収」で完全に一致する

無限に `End Turn` を重ねないよう、別途「先読みターン数」の上限が要る
（`max_continuation_steps` と同じ構造）。ブランチ数は増えるが、
`End Turn` は 1 行動なので増分は限定的。

`--turn-aligned-leaves` は既定 off のまま残す（実装とテストは正しく、
上記の形に進むときの土台になる）。**現時点で on にしてはいけない。**

## 11. ターン境界での leaf 採点（2026-08-20）

### 11.1 §10.7 を採らなかった理由

§10.7 は「`End Turn` を深さ予算から外し、その上で turn 整列する」案だった。これは
`[カード, カード]` と `[End Turn, カード, カード]` を「カード 2 枚消化」で並べる発想だが、
`lookahead_turns` という新しい予算軸と、その軸をまたいだ frontier を刈るための
cohort 分割が必要になる。どちらも `End Turn` を**探索する**ことを前提にした複雑さである。

`End Turn` を**探索しない**と決めると、両方とも不要になる。

### 11.2 何を実装したか

`BeamSearchConfig.turn_boundary_scoring`（既定 `False`）。

- 展開してよい行動を `TURN_INTERNAL_ACTION_TYPES`（`card` / `potion` / continuation）に
  限る。`End Turn` は展開されないので、**探索中にターン境界を跨ぐ行が生成されない**
- 各 leaf は「このターンにカードを k 枚打った状態」になる。root 自身も k=0 の leaf
  （＝「今すぐ End Turn」）として含める
- 全 leaf を強制 `End Turn` の先で採点する。root の leaf ではその `End Turn` が
  `root_action_id` になるので、`End Turn` は探索されずに選択可能なまま残る

これは `beam_searchable_action_types` とは**別の軸**である。後者は
`available_action_types(dto) <= allowed` という全称条件で「この state を探索してよいか」を
判定するため、そこから `system` を外すと `End Turn` が常に合法である以上すべての Combat
state が探索対象外になる（§P0 で踏んだのと同じ機構）。展開集合は分けて持つ必要がある。

§10 の `turn_aligned_leaves` は残す。遅れた leaf だけを `min(turn) + 1` へ持ち上げる
別の規則であり、同時指定は `__post_init__` が `ValueError` にする。

### 11.3 派生して必要になった 2 つの leaf 条件

1. **展開できる候補が 0 のノードは、そのノードだけ leaf にする。**
   従来は `_propose_frontier` が空を返すと `reason="no_candidates"` で**探索全体**を
   打ち切っていた。`End Turn` を展開しないと「打てるカードが無い」ノードが日常的に出る
   （実測: Combat 決定 2114 件のうち 308 件 = 14.6% が `End Turn` しか合法手を持たない）ため、
   全体打ち切りでは全行を失う
2. **カード自身がターンを終わらせた行は、そこで止める。**
   そのまま展開すると次ターンの内側に予算を使い、消したはずの半手ずれが戻る

### 11.4 §10.6 の残差がなぜ消えるか

A3 が効かなかったのは `[End Turn, カード]` という行が残り、ターン内で消費した
プレイヤー行動数が揃わなかったためだった。`End Turn` を展開しなければその行は
**生成されない**。整列対象は「ターン T でカード k 枚打った leaf」だけになり、
持ち上げ先は `root_turn + 1` の一点に定まる。§10.2 の「持ち上げは最大 1 ターン」
という hedge も、`min(turn) + 1` という顔ぶれ依存の目標も不要になる。

### 11.5 コスト

- `End Turn` を展開しないぶん、深さバッチのブランチは**減る**
- 追加は採点バッチのみ。ただし**バッチ上限だけでは足りない**（§11.7.1）。
  採点は branch slot を要求するのに、その時点で探索がほとんどを握っている。
  したがって採点の前に「採点対象の親以外」を release し、チャンク幅を
  `active branch capacity − 生存中の leaf 数` で決め、各チャンクの結果を受け取り次第
  そのチャンクのブランチと親を release する。採点済み state は二度と展開しないので、
  ブランチを保持し続ける理由が無い
- バッチが reject された場合は best effort で元のスコアが残る。
  `BeamSearchStats.leaves_turn_aligned` が 0 のままなら発火していない

### 11.6 検証

**単体**: `tests/decision/test_turn_boundary_scoring.py` 9 件。
「カードを多く打った行が有利に見えるのは攻撃が未払いだからで、払わせると順序が反転する」
という比較を fake client で再現し、off なら `card`(40.0)、on なら `end`(20.0) を選ぶこと、
決着後の順序がカード有利ならカードを選ぶこと、深さバッチに `End Turn` が一度も現れないこと、
`End Turn` しか合法手が無いノードで探索が中断しないこと、バッチ上限でチャンクされること、
採点バッチが reject されても元のスコアが残ることを assert する。全体 870 passed / 6 skipped。

**実機の受け入れ基準**（floor 到達数では判定しない。n=10 で CI ±2.8 のため）:

| 指標 | 現状（実測） | 期待 |
|---|---:|---|
| `End Turn` 選択かつ支払い可能なカードが手札に残る | **9.6%**（202 / 2114） | 低下 |
| `leaves_turn_aligned` | — | 非ゼロ（0 なら採点バッチが通っていない） |
| `branches_created` / 探索 | 平均 29.4 | 増えない |

現状値は `data/evaluation/detailed_logs/202608201812` の 11 戦から算出した。

## 11.7 測定結果（2026-08-20, 202608202142, n=10）

`--turn-boundary-scoring` を有効にした 10 戦。比較対象は同条件の
`data/evaluation/detailed_logs/202608201812`（`search_start` trace で
beam_width 8 / top_k 4 / max_depth 2 / learned pruner / `damage_race` の一致を確認済み）。

| 機構メトリクス | off | on | |
|---|---:|---:|---|
| **`End Turn` 選択かつ支払い可能なカードが残る** | 202 / 2114 = **9.6%** | 7 / 1382 = **0.5%** | 達成 |
| うち Energy 満タン | 27 | **0** | 達成 |
| `leaves_turn_aligned` | 0 | 7925（5.77 / 探索） | 発火 |
| `branches_created` / 探索 | 29.4 | 28.3 | 増えていない |
| `branches_faulted` | 16 | 2 | |

到達階層は completed 8 戦で平均 14.25（中央値 15.5 / 最大 23）。異常終了 2 戦の到達 floor
（31 と 15）を含めた 10 戦では平均 16.0。同条件の `dmgrace-verify` は 13.5（n=10）。
n=10 では CI が足りないため**判定には使わない**（§10.4）。

### 11.7.1 採点が 31% の探索で発火していなかった（修正済み）

```
leaves_turn_aligned = 0 の探索: 425 / 1374 (31%)
  うち reason=max_depth: 395
  それらの branches_created 平均: 48.8  （発火した探索は 29.4）
```

root は必ず採点対象に入るので、`aligned = 0` は**採点バッチ全体が reject された**ことを
意味する。RL の `instance_whole_run.py` は `max_branches = 64`。探索が既に 49 本
握っている状態で採点バッチを足すと超える。実装は `RequestRejectedError` を best effort で
握りつぶすため、**無音で無効化**されていた（`--log-level WARNING` では INFO も出ない）。

**未修正のまま残っていたのが「leaf が最も多い探索」＝ズレの影響が最も大きい探索**なので、
上表の 0.5% も 14.25 も、本来の効果の一部しか反映していない。

修正は §11.5 のとおり。回帰テストは
`ActiveBranchCapacityTest.test_settling_frees_branches_before_asking_for_more`
（capacity を持つ fake Whole Run client を使い、修正前は `leaves_turn_aligned == 0` で
実際に落ちることを確認済み）。ログレベルも INFO から WARNING に上げた。

### 11.7.2 解決済み: `AllBranchesFaultedError` による異常終了 2 件

原因は STS2_RL 側にあった。生き残った探索に残っていた fault の中身:

```
"Illegal action: 6"  at Sts2Emulator.Api.GameInstance.Step(Int32 actionId)
  action_type=card, action_id=6 (BASH), combat_depth=1   ← DTO は legal と公開していた
```

Combat branch の replay が、親の opaque な `action_id` をそのまま step していた。
`action_id` はそれが来た決定でしか有効でないため、replay 先では別の行動を指す。
さらに `_COMBAT_ACTION_TYPES` が `choice_card` / `choice_confirm` / `choice_skip` を含む
——焚き火の強化プロンプトやカード報酬が publish するのはまさにこれ——ため、
非戦闘の選択画面が Combat replay 経路に入っていた。

STS2_RL 側で、`room_context.room_type == "CombatRoom"` を Combat view の条件に加え、
選んだ行動を意味キー（`legal_action_semantic_key_text`）＋ ordinal で持ち回って
replay 先で解決し直すよう修正した（同名カードは意味キーを共有し、card instance id は
publish されていないため ordinal で区別する）。

`End Turn` を候補から外したことでこれが致命化していた。従来は frontier に常に成功する
`End Turn` が残っていたが、候補が全部カードになったため 1 件の fault で frontier が全滅した。

202608210802 で `branches_faulted = 0` / `branches_out_of_scope = 0` を確認。

### 11.7.4 未解決だった残り（旧 11.7.2 の記録）

過去の同種評価（`rest-verify` / `route-verify` / `dmgrace-verify` / `p0-verify`）は
いずれも `runs_errored = 0` だったので、これは回帰である。

生き残った探索に残っていた fault の中身:

```
"Illegal action: 6"  at Sts2Emulator.Api.GameInstance.Step(Int32 actionId)
  action_type=card, action_id=6 (BASH), combat_depth=1   ← DTO は legal と公開していた
```

**Emulator が legal と公開したカードを拒否する**という既存の不具合（§10.5 でも
`Illegal action` が 3 件記録されている）。今回それが致命化したのは、`End Turn` を候補から
外したことで frontier が**全部カード**になり、常に成功する手が候補に居なくなったためと
考えられる。従来は `End Turn` が生き残って frontier を維持していた。

落ちた 2 戦は floor 31 / floor 15 まで進んだ時点で死んでいる（300 / 168 決定）。
弱くて落ちたのではないので、`runs_completed` だけで平均を取る現在の集計は**過小評価**。

想定される対処（未実装）:

- `turn_boundary_scoring` で frontier が全滅したとき、親が macro-resolved なら親を leaf と
  して扱って探索を継続する。ただし `AllBranchesFaultedError` は「静かに落とさない」ために
  意図的に入れられたもの（commit 19f87e2）なので、握りつぶすのではなく fault を数えたまま
  継続する形にする
- 根本原因（Emulator の legal/illegal 不一致）は STS2_Emulator 側の調査が要る

### 11.7.3 解決済み: 落ちた決定の `score_trace` が失われる

`_TrackingCombatDecisionEngine.decide()` は `super().decide()` が**返ってから**
detailed log を書いていた。したがって `decide()` が送出すると、
**その決定の trace だけが記録されない**。異常終了した 2 戦がどちらも
`search_start` と `search_end` の件数が一致していたのはこのためで、
落ちた探索は両方に寄与していない。原因究明に最も必要な 1 件が、
構造的に必ず失われる形だった。

`decide()` を try/except で包み、送出時に

- `event="decision_failed"`（例外の型・メッセージ、分かる場合は root DTO）
- それまでに収集済みの `score_trace`（`decision_source="failed"`）

を書いてから再送出するようにした。`decision` 引数が無い呼び出し
（`decide_and_commit` の初回）では `decision_point_id` を trace の
`root_decision_point_id` から拾う。盤面自体は `selection` イベントが
リクエスト時点で書かれているので同じストリームから復元できる。

**テスト**: `tests/runner/test_failed_decision_logging.py` 5 件。
`decide()` が送出するケースで trace と `decision_failed` が書かれること、
`decision` 引数が無い場合の decision_point_id フォールバック、
trace が 1 件も無い時点での送出、および成功パスのラベル付けを固定する。
**try/except を外すと 4 件が実際に落ちる**ことを確認済み。

## 11.8 採点を ply ごとに行う（2026-08-21）

### 11.8.1 何が起きていたか

§11.7.1 の capacity 修正後も `leaves_turn_aligned == 0` は 30.9% → 25.8% にしか下がらなかった。
深さ別に見ると原因が一目で分かる（202608210802, 1850 探索）。

| `depths_completed` | 探索数 | `aligned == 0` | 率 |
|---:|---:|---:|---:|
| 2 | 524 | 0 | **0%** |
| 3 | 103 | 45 | 44% |
| 4 | 419 | 262 | **63%** |
| 5 | 150 | 115 | **77%** |
| 6+ | 47 | 30 | 64% |

`_release_unneeded_whole_run_branches` の `keep_ids` は **`beam` と `waiting_stable` だけ**で、
`finished` を含まない。この release はループ本体の末尾にあるので、
**ある ply で leaf になったノードは、その ply の終わりに release される**。
一方 leaf の採点は全 ply が終わったあとに一括で行っていた。

2 ply の探索だけ 100% 成功していたのは、leaf が全部「最後の ply」で生まれ、
最後の ply は `break` して release 呼び出しに到達しないため。3 ply 以上では
最初の ply の leaf が既に消えている。

失敗が「一部」ではなく **きれいに 0** になるのは、サーバが
**存在しない親 branch を含むバッチを丸ごと reject する**ため。root branch は決して
release されないので、個別 fault なら root の分だけは通って `aligned >= 1` になるはずだった。
§11.7.1 で capacity と結論したのは、この all-or-nothing を見落としたことによる誤診である
（capacity も実在の一因ではあり、修正で 48.8 vs 29.4 だった branch 数の差が
43.4 vs 41.0 まで縮んでいる）。

### 11.8.2 修正

採点を **leaf が生まれた ply の中で、release の前に**行う。

- `_settle_pending_leaves()` を、ループ内の 2 つの release 呼び出しの直前と、ループ後に呼ぶ。
  `settled_branch_ids` で二重採点を防ぐ
- ループ後の呼び出しは、最後の ply の leaf（`break` により release されていない）と、
  `waiting_stable` / `beam` の残りを拾う
- root は `_settle_root_as_a_leaf()` が引き続きループ後に処理する。root branch は
  release されないため安全
- `_settle_batch()` は、サーバの batch 上限に加えて **生存中ブランチ数から算出した
  capacity の余地**でチャンク分割し、各チャンクの DTO を受け取り次第そのチャンクの
  ブランチを release する。採点済み state は二度と展開しないので保持する理由が無く、
  返した slot が次のチャンクの余地になる

leaf の保持が 1 ply 内に収まるため、`finished` を release 対象から外す必要がなく、
`_fit_whole_run_frontier_to_active_capacity` の capacity 会計も変えずに済む。

コストは `emulate_actions` が探索あたり ply 数ぶん増える（幅と深さは不変）。

### 11.8.3 検証

`ReleasedLeafTest.test_a_leaf_is_settled_before_its_branch_is_released`。
コストの違う 2 枚のカードで、行ごとにエネルギーが尽きる ply をずらし、
release 済みブランチを親に持つバッチを reject する fake Whole Run client を使う。
**per-ply 採点を無効にすると、実運用と同じ WARNING を出して実際に落ちる**ことを確認済み。

`test_end_turn_is_never_expanded_during_the_search` は、採点バッチが ply ごとに
挟まるようになったため「各バッチは全 End Turn か End Turn を含まないかのどちらかで、
混在しない」というより強い不変条件に書き換えた。

全体 872 passed / 6 skipped。

### 11.8.4 次の測定

同条件でもう 1 回。受け入れ基準は `leaves_turn_aligned == 0` の探索が
（`not_beam_searchable` など採点対象が無いケースを除いて）ほぼ消えること。
効いていなかった 26% は **3 ply 以上＝最も深く leaf が多い探索**なので、
202608210802 の「0.2%」「平均到達階層 17.56」もまだ完全な状態の数字ではない。

