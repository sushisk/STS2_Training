# Whole Run 到達階層 改善計画

`docs/whole_run_status_20260820.md` の続きとして、`data/evaluation/score_logs/topk8-retry-00..04`
の全 score ログ（251 探索 / 2519 イベント）と実装を突き合わせた結果と、到達階層を上げるための
計画を示す。

## 0. 結論

Whole Run 評価では、**beam search が「対象選択を伴うカード」をすべて黙って捨てている**。
これは学習の質の問題ではなく、`floor_reach_eval` の engine 構築の設定漏れ（1 箇所）である。

- 全 251 探索のうち **178 回（71%）** で、root 候補の一部がフロンティアに到達せずに消えている。
- **113 回（45%）** では `targetType=AnyEnemy` の候補が**全滅**しており、その決定で
  エージェントは「攻撃する」という選択肢を一切評価できていない。
- 結果として、`SLIMED`（何もしないステータスカード）を最善手として 17 回選択している。

前回ドキュメントの「pruner による攻撃行動の除外は確認されず、`kept=true` だった」という観察は
正しいが、結論は逆で、**攻撃ノードは pruner に届く前に消えている**。同様に「候補漏れは
`top_k_actions=8` でほぼ解消した」も、policy 提案段階では解消しているが、
**提案の後段で消えている**ため実効的な漏れは残っている。

## 1. 根本原因

### 1.1 beam の意味スコープが狭いまま渡っている

`BeamSearchConfig.beam_searchable_action_types` の既定値は
`{"system", "card", "potion"}` で、continuation（`choice_target` / `choice_card` /
`choice_confirm` / `choice_skip`）を**含まない**（`src/sts2_training/decision/beam_search.py:81`）。

`CombatDecisionEngine` は「`beam_config` を明示的に渡された場合、その
`beam_searchable_action_types` を尊重する」設計になっている
（`src/sts2_training/decision/engine.py:100`）。
そのため runner 側で広げ直す必要があり、`runner/episode.py` はこれを
`_runner_mode_config()` で `COMBAT_BEAM_ACTION_TYPES` に広げている
（`src/sts2_training/runner/episode.py:74`, `:119`）。

**`runner/floor_reach_eval.py` の `_build_engine` はこの広げ直しを行っていない。**
`resolve_search_mode()` の返す既定スコープのまま `CombatDecisionEngine` に渡している
（`src/sts2_training/runner/floor_reach_eval.py:221`）。

同じ漏れが他の runner にもある。

| 構築箇所 | スコープ | 状態 |
|---|---|---|
| `runner/episode.py:120`（oracle collection が使用） | `COMBAT_BEAM_ACTION_TYPES` | 正常 |
| `runner/floor_reach_eval.py:222`, `:233`（**Whole Run 評価**） | `{system,card,potion}` | **欠陥** |
| `runner/self_play.py:209` | `{system,card,potion}` | 欠陥 |
| `runner/stable_pruner_ab.py:480` | 呼び出し元の config 次第 | 要確認 |

Oracle collection は `build_engine` 経由なので、**学習データ自体は正常**である。
壊れているのは評価経路（と self-play / pruner A/B）だけ。

### 1.2 スコープ外ノードが無言で捨てられる

`BeamSearchEngine._score_frontier` は、emulate 済みで正常に解決したブランチでも、
DTO がスコープ外だと `continue` で捨てる（`src/sts2_training/decision/beam_search.py:1016`）。

```python
if _is_whole_run_unresolved_out_of_scope(self._client, dto, cfg.beam_searchable_action_types):
    continue
```

このとき trace イベントも fault カウントもログも出ない。`branches_faulted` は 0 のままなので、
既存の branch-fault 計測（#72 / #74 / #79）では検出できない。

### 1.3 Emulator 側の仕様との組み合わせ

`Sts2Emulator/Api/GameInstance.cs:4082` — `TargetType.AnyEnemy` のカードは、
**生存敵が 2 体以上のときだけ** `_pendingTarget` を発行し、boundary が `pending_choice` になる。
1 体なら自動で対象が決まり `stable` のままになる。

したがって 1.1 + 1.2 の帰結は、

> **敵が 2 体以上いる戦闘では、攻撃カードと対象指定ポーションが探索から完全に消える。**

Act 1 は複数敵戦（Louse ×2、Slime 分裂、Gremlin Gang、Sentries など）が多く、
到達階層 3〜8 で全敗している現状と完全に整合する。

### 1.4 実測値（251 探索）

root 候補のうちフロンティアに到達しなかった割合。

| ラベル | 落ちた回数 | 残った回数 | 消失率 |
|---|---:|---:|---:|
| `POWER_POTION` | 70 | 0 | **100%** |
| `ATTACK_POTION` | 10 | 0 | **100%** |
| `NEOWS_FURY` | 19 | 2 | 90% |
| `GIANT_ROCK` | 17 | 5 | 77% |
| `HEADBUTT` | 14 | 5 | 74% |
| `TAUNT` | 14 | 8 | 64% |
| `MOLTEN_FIST` | 16 | 20 | 44% |
| `STRIKE_IRONCLAD` | **157** | 227 | **41%** |
| `FLASH_OF_STEEL` | 6 | 9 | 40% |
| `BASH` | 17 | 27 | 39% |
| `DEFEND_IRONCLAD` / `SLIMED` / `SHRUG_IT_OFF` / `BATTLE_TRANCE` ほか自己対象 | 0 | — | **0%** |

消失したのは `targetType` が `AnyEnemy` のカードと、mid-effect choice を開くポーションのみ。
自己対象・無対象のカードは 1 件も落ちていない。

具体例（search `15cb97b5…`、root に `STRIKE ×2`, `SLIMED ×2`, `End Turn`）:

- policy 順位: `STRIKE`(0), `STRIKE`(1), `End Turn`(2), `SLIMED`(3), `SLIMED`(4)
- 5 ブランチすべて emulate 成功・fault 0
- 深さ 1 の stable frontier に入ったのは `End Turn`, `SLIMED`, `SLIMED` の **3 件のみ**
- 最終選択 = `SLIMED`（best_value −19.5）

## 2. 副次的に確認された問題

### 2.1 価値関数のホライズン非対称（P1）

深さ 1 ノードの value 平均:

| ノード種別 | 件数 | 平均 value | 中央値 |
|---|---:|---:|---:|
| `End Turn` | 225 | **−16.53** | −16.29 |
| カード | 778 | −4.09 | −1.00 |

`End Turn` 側だけが「敵の攻撃を受けた後」の状態で評価されるため、常に約 12 点不利になる。
探索は同一ターン内で止まるカード枝と、ターンをまたぐ `End Turn` 枝を**同じ尺度で比較している**。

このバイアスがあると、`SLIMED` のような無意味なカードでもエネルギーを捨てて
`End Turn` を先送りするほうが「得」に見える。1.1 を直しても、この歪みは残る。

加えて `HeuristicValueFunction` の重みは block を二重計上している
（`decision/value.py:28`、`decision/combat_observation.py:172`）。

- `predicted_incoming_damage = max(0, 敵の攻撃予告 − block)`（重み −1.0）
- `player_block`（重み +0.5）
- `player_hp_ratio`（重み 40.0） → maxHp 80 なら **1 HP = 0.5 点**

つまり有効な block 1 点 = 1.5 点 ≒ **HP 3 点分**に評価され、余剰 block も HP 1 点分の価値を持つ。

### 2.2 評価中に ε-greedy 探索が有効（P1）

`floor_reach_eval._build_engine` は `HeuristicCombatSelector(random.Random(seed))` を作る
（`src/sts2_training/runner/floor_reach_eval.py:219`）。`epsilon` の既定値は **0.1**
（`src/sts2_training/selection/heuristic_selector.py:64`）。
これはデータ収集用の探索であり、評価では 0 にすべき。map_room 選択とカード fallback の
10% が一様ランダムになっている（1 run あたり `heuristic_fallback` は 4〜11 回）。

### 2.3 探索深さ 2 の不足（P2）

`max_depth=2` はターンの前半 2 手しか見えない。continuation は `combat_depth` を
消費しない設計なので、1.1 修正後は深さを上げてもブランチ数の増加は線形に近い。
「攻撃 3 枚で敵を倒し切ってターンを飛ばす」といった手筋は深さ 2 では原理的に見えない。

### 2.4 戦闘外の意思決定（P2）

- `CardDataRewardCardSelectionPolicy` は**必ずカードを取る**（skip しない）。
  デッキ圧縮の概念がなく、`skada_score` はデッキ文脈非依存の大域事前分布。
  実際に `SLIMED` を含む肥大デッキが観測されている。
- `room_preference_scores` は `Treasure > RestSite > Shop > Monster > Unknown > Elite` の固定表。
- rest site / shop の選択は `_choose()`（一様ランダム）に落ちる。

## 3. 計画

### P0 — 探索スコープの修復（最優先） — **実装済み（2026-08-20）**

実装内容は「6. P0 実装記録」を参照。以下は当初の計画項目。


1. **`runner/floor_reach_eval.py` の `_build_engine` を修正**
   `runner/episode.py` の `_runner_mode_config()` を共有ヘルパとして切り出し、
   `_build_engine` の両分岐で適用する。
   併せて `runner/self_play.py:209`、`runner/stable_pruner_ab.py:_default_engine_factory`
   にも同じ処理を適用する。
2. **恒久的な再発防止**
   - (a) `BeamSearchConfig.beam_searchable_action_types` の既定値を
     `COMBAT_BEAM_ACTION_TYPES` に変更する（罠そのものを消す。テスト影響あり）。
   - (b) 既定値は据え置き、runner 側の適用を 1 関数に集約し、
     「runner が engine を作るときは必ずこれを通す」ことを test で強制する。

   推奨は (b) を先に、落ち着いてから (a) を検討。
3. **観測性の追加（必須）**
   `_score_frontier` のスコープ外 `continue` に
   - `BeamSearchStats.branches_out_of_scope` カウンタ
   - `SearchTraceEvent`（`out_of_scope_drop`: branch_id / root_action_id / boundary /
     観測された action_types）

   を追加する。今回の欠陥は「fault ではないので既存計測に映らない」ことで
   長期間気付かれなかった。
4. **回帰テスト**
   - `floor_reach_eval` / `self_play` が構築した engine の
     `beam_search.config.beam_searchable_action_types == COMBAT_BEAM_ACTION_TYPES` を assert。
   - fake client で「敵 2 体 + `AnyEnemy` カード」を与え、root 候補がフロンティアに
     残ることを assert（`tests/decision/test_whole_run_combat_beam.py` に追加）。

**期待効果**: 複数敵戦で攻撃が可能になる。45% の決定で「攻撃という選択肢自体が無い」状態が解消する。

### P0.5 — 評価条件の是正（工数ほぼゼロ）

5. `floor_reach_eval` の fallback selector を `HeuristicCombatSelector(rng, epsilon=0.0)` にする。
   探索を残したい用途のために `--eval-epsilon`（既定 0.0）を CLI に足す。

### P1 — 価値関数の較正（P0 の効果を測ってから）

6. **ターン境界の非対称を解消する。**
   - (a) `player_hp_ratio` / `player_block` / `predicted_incoming_damage` の 3 項を、
     `effective_hp = hp − max(0, 敵攻撃予告 − block)` の 1 項（`effective_hp / maxHp`）に
     置き換え、二重計上をやめる。
   - (b) 葉の評価をターン整列させる（すべての葉を「自ターン開始時」で評価する）。

   まず (a) を実施。(b) は探索構造の変更を伴うため後段。
7. 既存の value 学習データに対してオフラインで (a) の重みを再フィットし、
   `HeuristicValueFunction` と `LinearValueModel` の順位相関を比較する。
8. **学習 board score の再評価。** 現状 `--board-score heuristic` で運用しているが、
   これは P0 の欠陥がある状態での比較結果に基づく判断である。P0 修正後に
   `learned` / `heuristic` を同条件で取り直す。

### P2 — 探索予算と戦闘外方策

9. P0 修正後に `--beam-depth 2 / 3 / 4` を同条件で比較。
   continuation が `combat_depth` を消費しない点を踏まえ、深さ 3 を第一候補とする。
   1 run の実測は 47〜150 秒なので、深さを上げた場合の実行時間を先に 1 run で測る。
10. 報酬カードに **skip** を含める。最小構成として
    「デッキ枚数が閾値超 かつ `skada_score` が閾値未満なら skip」。
11. rest site（休息 / 強化）と shop の選択に heuristic を入れる（現状は一様ランダム）。
12. `room_preference_scores` の重みを、到達階層を目的関数として見直す。

## 4. 検証プロトコル

現行の 5 戦は分散が大きく（母標準偏差 1.72）、判断材料にならない。

- **n = 30 / 条件**、`--ascension 0`、`IRONCLAD`、concurrency 1（サーバは
  `GameInstance` 共有のため並列不可）。1 run 約 60〜150 秒 → 1 条件あたり 40〜75 分。
- 一括実行がタイムアウトする問題は、`evaluate_whole_run.py` 側に
  「run ごとに逐次 JSON を書き出す」オプションを足すのが安全
  （現状 `topk8-20260820` は 5 戦分の結果を落としている）。
- 比較は独立 run の集計統計で行う（同一 seed の前後比較は 1 手違えば軌跡全体が変わるため無意味。
  `floor_reach_eval` の docstring も同じことを述べている）。
- **P0 修正の受け入れ基準（score ログで検証）**
  - スコープ外 drop = 0
  - `targetType=AnyEnemy` の候補消失率 = 0%
  - `SLIMED` が最善手に選ばれる回数 = 0

  これらは到達階層とは独立に、修正直後の 1〜2 run で即座に確認できる。

## 5. 実施順序

```
P0 (1-4)  →  1 run で score ログ検証（drop=0）  →  P0.5 (5)
   →  n=30 ベースライン再取得（これが正しい出発点）
   →  P1 (6-8)  →  n=30
   →  P2 (9-12) →  n=30
```

P0 修正前のすべての測定値（`top_k_actions=4` vs `8` の比較を含む）は、
「攻撃行動の 41〜100% が探索から欠落した状態」での測定であり、
モデル品質の指標としては使えない。P0 修正後のベースラインを取り直すまで、
学習側のチューニングは保留するのが妥当。

## 6. P0 実装記録（2026-08-20）

### 6.1 変更内容

**scope の一元化**

- 新規 `src/sts2_training/runner/beam_scope.py` に `runner_combat_beam_config()` を追加。
  preset の budget（depth / width / top-k / time budget）はそのままに、
  `beam_searchable_action_types` だけを `COMBAT_BEAM_ACTION_TYPES` に広げる。
- `runner/episode.py` の private `_runner_mode_config()` を削除し、この共有ヘルパに置換。
- `runner/floor_reach_eval.py` の `_build_engine`（**本欠陥の発生箇所**）を経由させた。
- `runner/self_play.py` の engine 構築を経由させた。
- `runner/stable_pruner_ab.py` の `_cli_beam_config()` を、重複していた同じ `replace()` から
  共有ヘルパ呼び出しに置換。

named/default preset には helper を適用し、呼び出し側が明示した `BeamSearchConfig` は
semantic scope を含めて保持する。これにより、runner のCombat用presetだけを安全に広げ、
低レベルAPIの明示的なscope指定は上書きしない。

`stable_pruner_ab._default_engine_factory` は呼び出し元の明示 config を尊重する既存挙動のまま
（「明示的な config は authoritative」という `CombatDecisionEngine` の契約に従う）。

**観測性**

- `BeamSearchStats.branches_out_of_scope` を追加。`branches_faulted` とは意図的に別カウンタで、
  非ゼロは transport/emulator の失敗ではなく設定の誤りを意味する。
- `SearchTraceEnd.branches_out_of_scope` を追加（score ログの `search_end` に出る）。
- `OutOfScopeDropTrace`（`event_type="out_of_scope_drop"`）を追加。
  `boundary` / `observed_action_types` / `allowed_action_types` を含むため、
  trace だけを見れば「どの admission 規則で落ちたか」が分かる。
- 該当 branch について WARNING ログを出力。
- `BeamSearchEngine._score_frontier` の戻り値は 6-tuple → 7-tuple。
  `oracle_search.py` / `oracle_value_logging.py` の override を追従させた
  （どちらも先頭 2 要素しか使っていないため index 参照に変更）。

**テスト**

- `tests/decision/test_beam_out_of_scope_drop.py`（新規、4 件）
  fake client で「`strike` が `choice_target` continuation を返す」状況を作り、
  - 完全 scope: `best_root_action_id == "strike"`、`branches_out_of_scope == 0`
  - 狭い scope: `best_root_action_id == "defend"`、`branches_out_of_scope == 1`、
    `branches_faulted == 0`、`OutOfScopeDropTrace` が 1 件で
    `boundary="pending_choice"` / `observed_action_types=("choice_target",)`
- `tests/runner/test_runner_beam_scope.py`（新規、6 件）
  `episode.build_engine` / `floor_reach_eval._build_engine`（4 search mode + depth override）/
  `self_play._run_one` / `stable_pruner_ab._cli_beam_config` の全 entry point が
  `COMBAT_BEAM_ACTION_TYPES` を持つことを assert。
- 既存の `_score_frontier` 直呼び出しテスト 4 ファイルを 7-tuple に追従。
- 全体: 768 passed / 6 skipped（skip は `STS2_RL_ROOT` 未設定の paired RL テスト）。

**検証ツール**

- `tools/check_score_log_scope.py` を追加。score ログを読み、
  root 候補のうち depth-1 stable frontier にも continuation 展開にも現れなかったものを集計する。
  欠落または `out_of_scope_drop` があれば exit code 1。

```
python tools/check_score_log_scope.py data/evaluation/score_logs/<dir>
```

修正前のログに対して実行すると、§1.4 の数字を再現する（`searches_with_missing_candidates` 178/251、
`searches_with_all_targeted_candidates_missing` 113、exit 1）。修正後は 0 / 0 / exit 0 になるはず。

**ドキュメント**

- `docs/02_decision_core.md` に §5.1「`beam_searchable_action_types` は runner が広げる」を追加。

### 6.2 次のステップ

1. Emulator サーバを起動し、score ログ有効で 1 run 実行する。

   ```
   python tools/evaluate_whole_run.py --character-id IRONCLAD --num-runs 1 \
       --models learned --search-modes standard --board-score heuristic \
       --beam-depth 2 --beam-width 8 --top-k-actions 8 \
       --output-dir data/evaluation/whole_run/p0-verify \
       --detailed-log-dir data/evaluation/detailed_logs/p0-verify
   ```

2. 受け入れ判定。

   ```
   python tools/check_score_log_scope.py data/evaluation/detailed_logs/p0-verify
   ```

   `searches_with_missing_candidates == 0`、`out_of_scope_drops == 0`、exit 0 であること。

3. 通ったら P0.5（eval 中の ε を 0 に）を入れ、n=30 のベースラインを取り直す。

### 6.3 P0 検証結果（2026-08-20）

- `top_k_actions=8` の1戦検証: 到達階層 9、エラーなし。
- `top_k_actions=4` の5戦検証: 到達階層 `11, 5, 5, 8, 6`、平均 7.0、分散 5.2、エラーなし。
- いずれも全戦の最終結果は敗北。

P0修正前の `top_k_actions=4`・5戦平均5.0と比べると改善しているが、seedが異なるため、
P0の効果を確定するには同一条件で追加の独立runが必要である。scope counterの正式な受入判定は、
scoreログを伴う検証で `check_score_log_scope.py` を実行して行う。
