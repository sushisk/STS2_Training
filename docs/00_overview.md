# STS2_Training ドキュメント概要

## 0. 文章の目的

この文書は、`STS2_Training` の新しいドキュメントセットの入口である。対象は `src/sts2_training/` の実装であり、Combat 意思決定モデルとその学習・評価・可視化パイプラインを、実際のコード構造に沿って読み進められるようにする。

## 1. 概要

`STS2_Training` は、 sibling repo である `STS2_RL` / `STS2_Emulator` の Training API に接続し、Slay the Spire 2 の意思決定を収集・探索・学習する Python パッケージである。中心は Combat の意思決定で、`PolicyModel`、`ValueModel`、`StableFrontierPruner` を組み合わせて、Beam Search による行動選択と教師データ生成を行う。

この repo は大きく分けて、API 接続、Combat decision stack、stable frontier pruning、Budgeted Oracle、Run-state board evaluation、非 Combat heuristic selection、runner CLI、live visualizer から成る。外部 wire protocol の版管理文書である `wire_contract_v0.7.md` と `wire_contract_v0.8.md` はこの rewrite の対象外であり、引き続き外部共有仕様として扱う。

## 2. Architecture

| 読みたいこと | 参照 |
|---|---|
| Training API/TCP 接続、DTO v0.8、retry/epoch 処理 | [01_api.md](01_api.md) |
| Combat Beam Search、Policy/Value、決定エンジン、score 用語 | [02_decision_core.md](02_decision_core.md) |
| StableFrontierPruner、learned pruner、feature/RL/training data contract | [03_stable_pruner.md](03_stable_pruner.md) |
| Budgeted Oracle 収集、Oracle JSONL、scenario harvesting | [04_oracle.md](04_oracle.md) |
| Run-state / deck 評価モデル、カード特徴、教師データ | [05_board_eval.md](05_board_eval.md) |
| reward/map/event/card choice など非 Combat heuristic selector | [06_selection.md](06_selection.md) |
| 実行用 CLI、self-play、A/B、floor reach、episode runner | [07_runner_cli.md](07_runner_cli.md) |
| JSONL replay/live dashboard、HTTP server、browser DTO | [08_visualizer.md](08_visualizer.md) |

## 3. API

この overview 自体に公開 API はない。実装で直接呼ぶ主要入口は、各 module doc の `## 3. API` を参照する。

## 4. 使用例

代表的な読み順は次のとおりである。

```text
最初に全体像を掴む:
  00_overview.md -> 01_api.md -> 02_decision_core.md

Oracle collection を動かす:
  04_oracle.md -> 07_runner_cli.md -> 03_stable_pruner.md

learned stable pruner を評価する:
  03_stable_pruner.md -> 07_runner_cli.md -> 08_visualizer.md

非 Combat の選択ロジックを見る:
  06_selection.md -> 05_board_eval.md
```

## 5. 補足説明

既存 docs のうち、古い設計案や段階的実装メモは、現在の実装と schema version がずれている箇所がある。特に Oracle record schema は現在 `ORACLE_RECORD_SCHEMA_VERSION = 6`、mask version は `1.2`、learned-pruner artifact schema は `2`、stable-pruner feature schema は `2` である。新しい文書では、古い番号や未実装の構想を runtime contract として扱わない。
