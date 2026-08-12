# Stable Pruner 利用ガイド

## 0. 文章の目的

### 0.1 この文書で扱うこと

この文書は、STS2_Training の combat search 学習機能を実際に運用するための手順書である。対象は、Budgeted Oracle による教師データ収集、supervised stable-pruner の学習・検証・実行、固定 seed A/B、on-policy RL fine-tuning までとする。

設計背景やアルゴリズムの詳細は `docs/combat_search_learning_plan.md`、`docs/stable_pruner_training.md`、`docs/stable_pruner_rl.md` を参照する。本書では「どの順番で、どのコマンドを実行し、何を次の入力にするか」を優先する。

### 0.2 この文書で扱わないこと

現在の learned pruner が制御するのは stable/resolved frontier の survivor 選択だけである。以下は本機能の学習対象ではない。

- `PolicyModel.top_k_actions`
- continuation の割り当て・順序
- parent expansion scheduling
- dynamic beam width / stopping
- Whole Run active-branch capacity
- broader `SearchController`

また、学習後 artifact の自動 promotion は行わない。最終判断には held-out Oracle 指標と real emulator A/B の両方を使う。

## 1. 概要

### 1.1 標準ワークフロー

推奨する処理順は次のとおりである。

```text
Combat scenario
  -> Budgeted Oracle JSONL を収集
  -> supervised artifact A を学習
  -> held-out Oracle validation
  -> ValueTopKPruner と fixed-seed A/B
  -> artifact A で stochastic RL trajectory を収集
  -> 1 batch REINFORCE update -> artifact B
  -> artifact B で fresh trajectory を再収集
  -> held-out validation / fixed-seed A/B を再実行
```

初期 supervised 学習を省略して RL から開始してはいけない。RL trajectory は生成時の behavior artifact SHA-256 に結び付いているため、更新後 artifact に古い trajectory を再利用することもできない。

### 1.2 現在のデータ契約

現在の主要 schema は次のとおりである。

| 対象 | version |
|---|---:|
| Oracle JSONL `combat_oracle_decision` | v4 |
| learned-pruner artifact | v2 |
| `StablePruneNodeView` | v1 |
| stable-pruner feature schema | v2 |
| RL trajectory `stable_pruner_rl_episode` | v2 |

Oracle v4 の `StablePruneTrace` は `selected_indices` を保存し、survivor の集合だけでなく実際の順序も replay できる。学習・更新コマンドは schema、teacher provenance、artifact SHA などを既定で厳密に検証する。

### 1.3 生成物の役割

運用ではファイルの役割を分ける。

- Oracle JSONL: supervised 教師データ。原則として episode/run ごとにファイル境界を保つ。
- learned-pruner JSON: runtime で読み込むモデル artifact。
- validation JSON: artifact を更新しない held-out 評価結果。
- A/B JSON: `ValueTopKPruner` と learned pruner の real emulator 比較結果。
- RL trajectory JSONL: 特定 artifact から生成された on-policy 更新用データ。

`stable-pruner-learn` は入力ログを一時的に正規化するが、元ログを書き換えない。`--output` が `--weights` または解決済み入力ログと同じ path を指す場合は fail closed する。

## 2. セットアップと Oracle データ収集

### 2.1 インストールと接続前提

リポジトリの source checkout で Python 3.11 以上を使用し、学習依存を含めてインストールする。

```bash
pip install -e ".[train]"
```

`[train]` は `numpy` / `scikit-learn` を追加する。runtime の linear pruner inference 自体は標準ライブラリのみで動作する。

Oracle collection、A/B、RL trajectory collection は STS2_RL / Emulator の Training API に接続する。CLI の既定値は `--host 127.0.0.1 --port 8765` である。必要なら各コマンドで `--host` / `--port` を上書きする。

### 2.2 Budgeted Oracle を収集する

scenario JSON を用意し、通常の runtime engine が行動を commit する状態分布上で wider/deeper Oracle を走らせる。

```bash
python -m sts2_training.runner.oracle_collection \
  --scenario data/scenarios/slime.json \
  --output data/combat_oracle/slime-001.jsonl \
  --search-mode standard \
  --oracle-beam-width 32 \
  --oracle-top-k 8 \
  --oracle-depth 4 \
  --target-beam-width 8
```

`--target-beam-width` は学習対象となる runtime/student K を表す。省略時は runtime search mode の beam width が使われる。Oracle 自体の beam width と混同しない。

root action は既定で exhaustive に評価する。`--policy-limited-root` を付けると policy-limited collection になり、未評価 legal action は censored / `no_target` 扱いになる。通常の教師データ収集では、明確な理由がない限り exhaustive root の既定値を維持する。

### 2.3 収集データを確認する

Oracle 出力は `combat_oracle_decision` の JSONL である。学習前に少なくとも次を確認する。

- `record_schema_version` が 4 である。
- `provenance` に teacher Policy / Value の識別情報がある。
- stable-prune trace に ordered `selected_indices` がある。
- `target_source` が `terminal` / `value_bootstrap` / `no_target` 等として記録され、`no_target` を負例として解釈していない。
- 異なる teacher 設定を意図せず同一 dataset に混在させていない。

`stable-pruner-learn` は `.jsonl`, `.json`, `.log`, `.txt` を入力として受け取り、directory は再帰的に探索する。ログ行に prefix/suffix が付いていても、内部で current Oracle / RL JSON record を抽出する。

## 3. Supervised 学習・評価・runtime 利用

### 3.1 初回 supervised artifact を作る

最短コマンドは次である。

```bash
stable-pruner-learn data/combat_oracle \
  --output tools/output/stable_pruner_weights.json
```

`--weights` が無い場合、既定では `--learn supervised --start fresh --data-mode auto-split` として動作する。明示する場合は次のように書ける。

```bash
stable-pruner-learn data/combat_oracle \
  --learn supervised \
  --start fresh \
  --data-mode auto-split \
  --val-fraction 0.1 \
  --test-fraction 0.1 \
  --seed 0 \
  --output tools/output/stable_pruner_weights.json
```

split は frontier 単位ではなく source JSONL file 単位で行う。そのため、収集段階から episode/run ごとにファイルを分けておく方が leakage を抑えやすい。

既定の target weight は terminal `1.0`、value bootstrap `0.5` である。`no_target` は frontier には残るが pairwise label には使われない。

### 3.2 新しい Oracle データで supervised resume する

既存 artifact を初期値として、新しい Oracle data をすべて update 用に使う場合は `train` mode を使用する。

```bash
stable-pruner-learn data/new_oracle_logs \
  --learn supervised \
  --start resume \
  --data-mode train \
  --weights tools/output/stable_pruner_weights.json \
  --output tools/output/stable_pruner_supervised_v2.json
```

`--output` は入力 `--weights` と別ファイルにする。teacher provenance は既定で一致を要求する。`--allow-mixed-teachers` / `--allow-teacher-mismatch` は意図した診断・実験時だけ明示的に使う。

resume 後は係数が変わるため、旧 artifact の metrics は current metrics として扱われず、再 validation が必要になる。

### 3.3 held-out Oracle で validation する

別に確保した Oracle dataset で artifact を更新せず評価する。

```bash
stable-pruner-learn data/combat_oracle_heldout \
  --learn supervised \
  --start resume \
  --data-mode validate \
  --weights tools/output/stable_pruner_weights.json \
  --output tools/output/stable_pruner_validation.json
```

`validate` は coefficient を変更せず、teacher provenance を既定で厳密に照合する。主に pairwise accuracy、label coverage、Recall@K、`ValueTopKPruner` との比較、fully-labeled frontier 上の regret/gap を確認する。

teacher-distillation 指標だけで gameplay 改善を確定してはいけない。次の real emulator A/B と組み合わせる。

### 3.4 runtime で learned pruner を使用する

生成 artifact は `LinearStableFrontierPruner` で読み込む。

```python
from sts2_training.decision import CombatDecisionEngine, LinearStableFrontierPruner

pruner = LinearStableFrontierPruner.from_weights_file(
    "tools/output/stable_pruner_weights.json"
)
engine = CombatDecisionEngine(client, stable_pruner=pruner)
```

learned pruner が変更するのは stable frontier の survivor selection のみであり、PolicyModel や beam width 等を自動変更しない。artifact SHA-256 由来の version が search trace に残るため、評価結果と artifact を対応付けられる。

### 3.5 fixed-seed A/B を実行する

promotion 判断前に、同じ scenario / seed / search config で `ValueTopKPruner` と比較する。

```bash
python -m sts2_training.runner.stable_pruner_ab \
  --scenario data/scenarios/slime.json \
  --weights tools/output/stable_pruner_weights.json \
  --seeds 101,102,103,104 \
  --search-mode standard \
  --output tools/output/stable_pruner_ab.json
```

見るべき項目は terminal outcome だけではない。nodes expanded、Beam search time、divergence、unknown/tie の扱いも確認する。複数 scenario をまとめる場合は `python -m sts2_training.runner.stable_pruner_ab_suite` と manifest を使う。

repo-local CI はコード・contract regression を確認するものであり、real Training/RL commit pair の gameplay 品質を保証するものではない。

## 4. RL fine-tuning と継続運用

### 4.1 現在の artifact で trajectory を収集する

supervised / current artifact を behavior policy として、real emulator 上で paired stochastic trajectory を収集する。

```bash
python -m sts2_training.runner.stable_pruner_rl \
  --scenario data/scenarios/slime.json \
  --weights tools/output/stable_pruner_weights.json \
  --seeds 101,102,103,104 \
  --temperature 1.0 \
  --output data/stable_pruner_rl/batch-a.jsonl
```

baseline arm は `ValueTopKPruner`、learned arm は指定 artifact を Plackett-Luce で stochastic 化した stable-pruner である。既定 reward は learned-minus-baseline の terminal outcome 差である。必要なら `--node-cost-weight` / `--beam-ms-cost-weight` を明示し、search cost penalty を追加する。

unknown outcome または実際の pruning choice が無い episode は RL batch から除外される。

### 4.2 1 batch だけ REINFORCE update する

収集した trajectory と、それを生成した**同一 artifact**を指定する。

```bash
stable-pruner-learn data/stable_pruner_rl/batch-a.jsonl \
  --learn rl \
  --start resume \
  --data-mode train \
  --weights tools/output/stable_pruner_weights.json \
  --output tools/output/stable_pruner_rl_v1.json
```

RL update は exactly one batch である。updater は behavior artifact SHA、artifact/node-view/feature schema、behavior score、sampled/returned indices、Plackett-Luce log-probability、temperature / sampler seed、paired reward の再計算整合を fail closed で検証する。

`RL + fresh` は禁止されている。RL は必ず既存 supervised/current artifact から開始する。

### 4.3 artifact 更新後は fresh trajectory を再収集する

上の update で artifact SHA が変わるため、`batch-a.jsonl` は新 artifact の on-policy data ではない。同じ trajectory を再利用せず、新 artifact で次の batch を収集する。

```bash
python -m sts2_training.runner.stable_pruner_rl \
  --scenario data/scenarios/slime.json \
  --weights tools/output/stable_pruner_rl_v1.json \
  --seeds 201,202,203,204 \
  --temperature 1.0 \
  --output data/stable_pruner_rl/batch-b.jsonl
```

次の update は `batch-b.jsonl` と `stable_pruner_rl_v1.json` の組み合わせで行う。この「collect -> one update -> fresh collect」の順序を崩さない。

### 4.4 更新 artifact を再評価する

各 supervised resume / RL update 後は、少なくとも次を再実行する。

1. held-out Oracle `--data-mode validate`
2. fixed-seed real emulator A/B
3. 必要に応じて multi-scenario A/B suite

positive training-batch reward、低い supervised loss、repo-local CI success のいずれも単独では promotion 条件にならない。比較する artifact、scenario、seed、teacher provenance、search config を固定し、品質と search cost の両方を確認してから runtime artifact を切り替える。
