# Whole Run 戦闘分岐の Combat Instance 委譲

## 0. 文章の目的

Whole Run 評価で戦闘中の `emulate_actions` が全 branch 失敗（`AllBranchesFaultedError`）に
なる問題について、原因・採った設計・実装結果・現時点の実測を記録する。

実装は STS2_RL 側（branch `agent/whole-run-combat-snapshot`、head `0cbb2f7`）にあり、
本文書はその**設計判断と Training 側から見た効果**を残す。RL 側の実装仕様書は
`STS2_RL/docs/whole_run_combat_snapshot_branching.md` が正本であり、手順ごとの詳細は
そちらを参照する。

## 1. 概要

**原因**: Whole Run の戦闘 branch は、worker プロセスで
`load_state` → `choose_room` → 部屋全体の `action_prefix` 再生を行っていた。
再生は**新しい RNG のもとで**走るため、root とは別のゲームになる。
戦闘の分岐は draw 順仮説に依存するので、これは必ず食い違う。

**採った設計**: 部屋の再生をやめ、`CombatInstance` が持つ復元セマンティクス
（stable アンカーの snapshot + そこからの `replay_prefix`）を worker に渡す。
戦闘の意味論を `CombatPhase` として切り出し、Whole Run は戦闘中それに委譲する。

**結果**: 6 条件の試走で 4 条件が完走した（修正前は 0）。
残る 2 条件は**別原因**の Emulator 側欠落で落ちている（§5）。
`AllBranchesFaultedError` は完全には解消していない。

## 2. Architecture

```
戦闘中の実行者と観測者
  WholeRunSession   保持する状態は _game のみ → 観測は陳腐化しない → 観測者
  LiveCombatSession _current_frame 等を保持    → 陳腐化する         → 実行者

  戦闘中に盤面を進めるのは CombatPhase だけ。Whole Run が進めると
  LiveCombatSession._is_still_current() が必ず false になり、次の commit が
  ResetFromScenario で共有 GameInstance を上書きしてランを消す。
```

```
分岐の流れ（委譲後）
  choose_room() → room_context.room_type == "CombatRoom"
    → enter_combat_phase()   RUN → TRANSFERRING → adopt（非破壊）→ COMBAT
    → emulate_actions        アンカー snapshot + rng_id を worker へ
    → commit_action          CombatPhase.commit_root_action
    → CombatCompletion 検出  → leave_combat_phase()  COMBAT → TRANSFERRING → RUN
```

実装は 7 手順に分けた。各手順の commit と全スイート結果は次のとおり。

| 手順 | 内容 | commit | pytest |
| --- | --- | --- | --- |
| S1 | `WholeRunSession` に combat snapshot の capture/restore | `0ba8384` | — |
| S2 | `GameAccess`（プロセス 1 個の `GameInstance` と lease） | `8aaa341` | 511 |
| S3 | 戦闘終了の引き渡しレコード / `whole_run_mode` フラグ | `6b420cc` | 516 |
| S4a | `CombatPhase` の切り出し（純粋な移動） | `f6f3d60` | 516 |
| S4b | adopted モードと「指せる／分岐できる」の分離 | `91a9cb4` | 522 |
| S5 | `enter_combat_phase` / `leave_combat_phase` のトランザクション化 | `8388363` | 529 |
| S6 | root commit の委譲 | `8021667` | 533 |
| S7 | branch トランザクションと分岐の委譲 | `e777c8f` | 538 |
| — | 戦闘終端 DTO の `outcome` 欠落と診断出力の修正 | `0cbb2f7` | 538 |

Training 側のコードは変更していない。委譲は RL 側で完結し、Training からは
同じ `emulate_actions` / `commit_action` 応答として見える。

## 3. API

Training 側で使うのは既存の評価 CLI のみ。RL サーバをワークツリーに向けて起動する。

`tools/evaluate_whole_run.py` の関連引数:

- `--rl-root PATH`: RL サーバを起動するリポジトリのパス
- `--start-rl-servers N`: 起動するサーバ数
- `--num-runs N` / `--max-decisions N`: ラン数と 1 ラン当たりの決定数上限
- `--turn-boundary-scoring`: End Turn を展開せず全 leaf をターン境界で整列する
- `--output-dir` / `--detailed-log-dir`: 評価レポートと詳細ログの出力先

戦闘分岐が失敗したときの原因は、`--output-dir` 直下の `rl-server-<port>.log` に
`[FAULT] branch=... phase=combat diagnostics=...` として残る。
ワイヤに出る応答はマスク済みで原因を含まないため、**根本原因はこのログでしか読めない**。

## 4. 使用例

短い試走（1 ラン、6 条件、約 10 分）:

```bash
python tools/evaluate_whole_run.py \
  --character-id IRONCLAD \
  --num-runs 1 \
  --max-decisions 250 \
  --turn-boundary-scoring \
  --start-rl-servers 1 \
  --rl-root "C:/STS2_RL-combat-snapshot" \
  --output-dir data/evaluation/whole_run_s7_trial3 \
  --detailed-log-dir data/evaluation/detailed_logs_s7_trial3
```

戦闘分岐の失敗原因を読む:

```bash
grep -ah "\[FAULT\]" data/evaluation/whole_run_s7_trial3/rl-server-*.log | head -1
```

## 5. 補足説明

### 5.1 到達 floor（2026-08-22 の試走、1 ラン / 条件、`--max-decisions 250`）

| 条件 | depth | width | top-k | 到達 floor |
| --- | --- | --- | --- | --- |
| baseline:standard | 2 | 8 | 4 | 11 |
| baseline:deep | 4 | 8 | 4 | **17** |
| baseline:wide | 2 | 16 | 6 | 13 |
| learned:standard | 2 | 8 | 4 | 7 |
| learned:deep | 4 | 8 | 4 | 未完走 |
| learned:wide | 2 | 16 | 6 | 未完走 |

**各条件 1 ランなので、条件間の優劣をこの数字から読んではならない。**
Whole Run の到達階層は seed 分散が大きく、比較には複数ランが要る。
この試走の目的は委譲経路が動くことの確認であり、性能測定ではない。

### 5.2 意図的に受け入れた制約

- **分岐は戦闘境界で終端する**: 戦闘 snapshot はラン位置を持たないため、戦闘を
  終わらせた branch はその先（報酬・地図・ラン終了）を生成できない。到達範囲の
  縮小だが、Training は `transition.kind == "combat_completed"` を terminal 扱いし、
  戦闘外へ出た branch を `branches_out_of_scope` として既に捨てているため、
  作れなくなるのは元々使っていないものだけである。

### 5.3 今後の課題

**1. RNG が関与しない枝展開の重複計算**

現状の beam は、ターン内で同一盤面に至る異なる行動順を別々に展開している。
RNG が絡まない範囲では、同じ盤面に到達する経路の探索結果は共有できるはずで、
ここを整理すれば同じ計算量でより深く読める。`--turn-boundary-scoring` で
leaf をターン境界に揃えたことが前提整備になっている。
[02_decision_core.md](02_decision_core.md) の beam 実装に対する変更となる。

**2. 戦闘開始時 pending choice**

`GAMBLING_CHIPS` のような開始時効果があると戦闘が `pending_choice` で始まる。
`pending_choice` の snapshot は復元できない（`unsupported_capture_boundary:published_target`）
ため、その時点で復元可能なアンカーが存在せず、**最初の `stable` に達するまで分岐できない**。
現状はその期間の分岐要求を明示的な理由付きで拒否し、Training 側の search trace に
`emulate_actions_rejected` として残るので**後から件数を数えられる**。
件数が問題になるなら、`CombatInstance` 側の `CombatStartReplayRoot` の機構を
Whole Run にも通す必要がある。

**3. `EYE_WITH_TEETH` の snapshot 復元拒否（Emulator 側）**

試走で残った 2 条件はこれで落ちている。

```
SnapshotRestoreRejectedException:
  unknown_monster_move_id:EYE_WITH_TEETH:REVIVE_MOVE
  at SnapshotRestorer.ResolveMoveState(MonsterModel monster, String moveId)
  at SnapshotRestorer.ApplyEnemyMoves()
```

`EYE_WITH_TEETH` の `REVIVE_MOVE` が snapshot 復元時のムーブ解決表に無く、
その敵を含む戦闘では**全 branch が同じ理由で失敗する**（同一 `snapshot_id`）。
委譲経路自体は正しく動いており、部屋再生をやめたことで露出した別原因である。
根本解決は `C:\STS2_Emulator`（C#）側の対応。RL 側で緩和するなら、復元できない
敵を含む戦闘では課題 2 と同じく分岐を拒否して件数を数える形になる。
