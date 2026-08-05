# Training最小設計 v0.1

## 0. 目的

本書は、RL–Training Communication API v0.5を利用するTraining側の最小構成を定義する。

最初の目標は、完了したRunの勝敗から状態価値を学習し、その価値を使ってTrainingがRLへBranch作成とroot Action確定を指示できるようにすることである。

初期版ではPolicy Model、深い探索、分散学習、Reward Shapingは実装しない。

---

## 1. 設計方針

Trainingは判断主体として、次を行う。

- RL APIを通じてEpisodeを開始・終了する
- Legal Actionから評価対象候補を選ぶ
- 必要なBranchだけをRLへ作成させる
- Branch到達状態をValue Modelで評価する
- 採用ActionをrootへCommitする
- rootのTrajectoryを保存する
- Run終了後の勝敗を教師値としてValue Modelを更新する

RLは状態遷移、Emulator、Branch、RNG Hypothesis、Worker、Snapshot、Replay、Leaseを管理する。Trainingはこれらを直接操作しない。

---

## 2. 最小アーキテクチャ

```text
Training Process
├─ TrainingApiClient
├─ EpisodeController
├─ CandidateSelector
├─ DecisionPolicy
├─ MaskedDtoEncoder
├─ ValueModel
├─ TrajectoryWriter
├─ ReplayBuffer
├─ Trainer
└─ CheckpointManager
        │
        │ RL–Training API v0.5
        ▼
RL Runtime Process
        ▼
Emulator Workers
```

初期版では、1つのTraining Processが1つのRL instanceを同期的に操作する。

学習とEpisode実行は同時並行にせず、Episode境界でModelを更新する。1 Episodeの途中でModel Versionを変更しない。

---

## 3. コンポーネント

### 3.1 `TrainingApiClient`

RL–Training API v0.5のClient。

担当：

- Request IDの発行
- Request／ResponseのValidation
- `instance_id`、`branch_id`、`decision_point_id`の保持
- Responseの`request_id`一致確認
- timeout時の安全な再送
- BranchのCancel／Release
- instance終了時のcleanup

同一処理の再送には同じ`request_id`を使用し、二重実行を防ぐ。

### 3.2 `EpisodeController`

1回のWhole Runまたは独立Combatを進行する。

担当：

- `start_instance`
- Decision Loop
- 候補Branchの作成
- ActionのCommit
- Trajectory記録
- terminal判定
- `close_instance`

通常時にRL側の`zero_index`は使用しない。全ActionはTrainingが明示的に指定する。

### 3.3 `CandidateSelector`

Legal ActionからSimulation対象を選ぶ。

初期仕様：

- Legal Action数が上限以下なら全件を選ぶ
- 上限を超える場合は決定論的に一部を抽出する
- 抽出乱数はTraining側のseedで管理する
- 同じ入力とseedから同じ候補集合を作る
- 候補の勝敗評価やHeuristic Scoreは持たない

初期上限：

- Combat：8候補
- その他：16候補

候補抽出はAction IDの単純な先頭固定ではなく、Action TypeまたはSemantic Groupごとに偏りを抑えて抽出する。詳細なCard Heuristicは導入しない。

### 3.4 `DecisionPolicy`

Branch評価からCommitするActionを決定する。

初期仕様：

1. 候補ごとに`emulate_action`を要求する
2. terminal Branchは実際の勝敗をScoreとする
3. 非terminal BranchはValue Modelの予測値をScoreとする
4. 最も高いScoreのActionを選択する
5. 探索用の確率で候補内からランダム選択する
6. 選択Actionを`commit_action`する
7. 作成したBranchをReleaseする

Branchがすべて失敗した場合は、TrainingがLegal Actionの先頭をFallbackとしてCommitし、理由をTrajectoryへ記録する。

### 3.5 `MaskedDtoEncoder`

`masked_emulator_dto`をValue Model入力へ変換する。

初期入力：

- Decision Type
- HP、Max HP、Gold、Energy、Block
- Floor、Act、Ascension
- Hand
- Draw／Discard／ExhaustのMultiset
- Deck
- Relic、Potion、Orb、Power
- Enemy状態とIntent
- Map上の公開情報
- 現在のEvent／Reward／Shop／Rest情報

可変長要素はID EmbeddingとPoolingで固定長へ変換する。

EncoderはHidden Informationを補完・推測しない。未知IDは`UNK`へ変換する。

### 3.6 `ValueModel`

状態から最終Run勝率を予測する。

```text
V(s) = P(run_win | masked_state=s)
```

初期構成：

- 共通のNumeric Feature層
- Card／Relic／Enemy等のEmbedding
- 可変長集合のMeanまたはSum Pooling
- Decision Type Embedding
- 小規模MLP
- 出力1 Logit

初期版ではPolicy Headを持たない。

### 3.7 `TrajectoryWriter`

root上で実際に進行したDecisionだけを正解Trajectoryとして保存する。

Branch到達状態は評価ログとして保存してよいが、終局まで継続していないBranchへ推定Labelを付けて学習データにはしない。

1 Decisionの記録項目：

- instance ID
- Episode内Step
- Decision Type
- masked Emulator DTO
- Legal Actions
- 候補Action IDs
- Branch評価結果
- 選択Action
- 選択確率
- Fallback有無
- Model Version
- terminal情報
- 最終Run結果

### 3.8 `ReplayBuffer`

完了Episodeだけを学習対象として保持する。

長いEpisodeが過剰に重くならないよう、次の順でSampleする。

1. Episodeを一様に選ぶ
2. そのEpisode内のDecisionを一様に選ぶ

または、各Decisionへ`1 / episode_length`のWeightを付ける。

### 3.9 `Trainer`

Value Modelを更新する。

初期教師値：

- Run勝利：`1`
- Run敗北：`0`

完了したroot Trajectory内の全状態へ、同じ最終結果を付与する。

初期Loss：

- Binary Cross Entropy with Logits

初期版では次を行わない。

- TD Learning
- Bootstrapping Target
- Branch予測値を教師にする自己蒸留
- Reward Shaping
- Policy Gradient

### 3.10 `CheckpointManager`

次を一体で保存する。

- Model State
- Optimizer State
- Encoder Vocabulary
- Training Config
- Training Step
- Episode Count
- Model Version
- 評価指標

一時ファイルへ保存してからAtomic Renameする。

Modelの読み替えはEpisode境界だけで行う。

---

## 4. 実行モード

### 4.1 Bootstrap Mode

Value Modelが未学習の期間。

- Branch評価を行わない
- TrainingがLegal ActionからランダムにActionを選ぶ
- root Trajectoryだけを収集する
- 一定数の完了Episode後に初回学習を行う

目的は、未学習Modelの出力で行動を固定してしまうことを防ぐことである。

### 4.2 Value-Guided Mode

初回Model作成後の通常モード。

- Candidateを選ぶ
- BranchをSimulationする
- Branch到達状態をValue Modelで評価する
- Valueと探索率に基づいてActionをCommitする
- Episode終了後にReplayへ追加する
- 指定間隔でModelを更新する

---

## 5. Decision Loop

```text
start_instance
    ↓
現在Decisionを取得
    ↓
CandidateSelectorで候補を選択
    ↓
候補ごとにemulate_action
    ↓
Branch状態をValue Modelで一括推論
    ↓
DecisionPolicyがActionを選択
    ↓
commit_action
    ↓
Branchをrelease
    ↓
Trajectoryへ記録
    ↓
terminalでなければ繰り返す
    ↓
最終勝敗を付与
    ↓
Replayへ保存
    ↓
必要なら学習・Checkpoint
    ↓
close_instance
```

推論は候補BranchをまとめてBatch化する。

---

## 6. RNGの利用

TrainingはBoundaryのCapabilityに応じて`rng_id`を指定する。

### Combat

Combat側のRNG Hypothesis機構を利用する。

初期版では、全候補を同じ`rng_id=1`で比較する。複数Hypothesis平均は後から追加する。

### Active Event

Active Event RNG Hypothesisを利用する。

同一候補比較では同じ正の`rng_id`を使用する。

### その他のWhole Run Boundary

Map、Reward、Shop、Rest等では、現在RL APIが許可する決定論的Simulation方式を利用する。

正の`rng_id`が未対応Boundaryでは要求しない。Capabilityまたは拒否契約に従う。

---

## 7. 学習データ

### 7.1 Raw Episode

監査可能なJSONLとして保存する。

```text
TrainingData/
└─ episodes/
   ├─ episode_000001.jsonl
   ├─ episode_000001.summary.json
   └─ ...
```

Episode完了前のファイルは学習対象にしない。

### 7.2 Dataset Split

Run Seedを基準にTrain／Validationへ分割する。

同じSeedのEpisodeが両方へ入らないようにする。

### 7.3 初期評価指標

- Validation Log Loss
- Brier Score
- ROC-AUC
- Calibration Error
- 実際のRun勝率
- Decision Type別の予測誤差

Model採用はRun勝率だけでなく、Validationの確率予測品質も確認する。

---

## 8. Fault処理

### API timeout

同じ`request_id`で再送し、二重実行を防ぐ。

### Branch fault

そのBranchを評価対象から除外する。ほかの候補は継続する。

### 全Branch失敗

TrainingがFallback Actionを選択してrootへCommitする。

### stale Decision

現在Decisionを再取得し、古い候補評価を破棄する。

### Training停止

既存BranchをCancel／Releaseし、`close_instance`を呼ぶ。

### Model／Checkpoint障害

最後に正常保存されたCheckpointへ戻す。RL状態とは独立して扱う。

---

## 9. 初期設定

以下は実装開始時の暫定値とする。

| 項目 | 初期値 |
|---|---:|
| Combat候補上限 | 8 |
| その他候補上限 | 16 |
| RNG Hypothesis数 | 1 |
| Bootstrap完了Episode数 | 100 |
| Batch Size | 256 |
| 学習間隔 | 10 Episode |
| 探索率 | 0.10 |
| Checkpoint間隔 | 100 Episode |
| Episode同時実行数 | 1 |

すべてConfigから変更可能にする。

---

## 10. ディレクトリ案

```text
Training/
├─ api/
│  ├─ client.py
│  └─ protocol.py
├─ controller/
│  └─ episode_controller.py
├─ decision/
│  ├─ candidate_selector.py
│  └─ value_policy.py
├─ model/
│  ├─ encoder.py
│  └─ value_model.py
├─ data/
│  ├─ trajectory.py
│  └─ replay_buffer.py
├─ train/
│  ├─ trainer.py
│  └─ metrics.py
├─ checkpoint/
│  └─ manager.py
├─ config.py
├─ main.py
└─ tests/
```

---

## 11. 初期版の対象外

- Policy Model
- MCTS／Beam Search
- 複数Decision先のTraining主導探索
- 複数Actorの並列収集
- Distributed Training
- Prioritized Replay
- TD Target
- Reward Shaping
- Map／Encounter／Boss／AncientのRNG Hypothesis
- Map Boundary越しのBranch再基準化
- Branch状態への疑似Label付与

---

## 12. 受け入れ条件

1. Training Processでpythonnet／CLRを初期化しない。
2. RL API v0.5だけを通じてEpisodeを進行できる。
3. Bootstrap Modeで完了Episodeを収集できる。
4. 完了EpisodeからValue Modelを学習できる。
5. Checkpointから学習と推論を再開できる。
6. Value-Guided ModeでBranchを一括推論し、ActionをCommitできる。
7. Branch評価によってrootが変化しない。
8. Branch到達状態を真の教師Labelとして誤使用しない。
9. Hidden InformationをModel入力へ含めない。
10. 同一Seed、同一Training乱数、同一Model Versionで行動列を再現できる。
11. Episode終了時にBranchとinstanceを完全に解放できる。
12. CombatとWhole Runの双方でE2E Smoke Testが成功する。
