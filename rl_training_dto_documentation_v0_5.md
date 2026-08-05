# RL–Training Communication DTO Documentation v0.5

## 0. この文書の目的

本書は、TrainingとRL Runtimeの間で使用する通信DTOを定義する。

TrainingはAction選択とBranch作成を指示し、RLはrootの進行、Branch Simulation、Emulator操作、RNG Hypothesis、Worker、Snapshot、Replay、Leaseを管理する。

Emulator DTOは公開可能な情報を基本的にそのまま利用し、Hidden InformationとRL内部情報だけを削除またはマスクする。

---

## 1. 概要

- Trainingを判断主体、RLを実行主体とする。
- RLはTrainingから指定されたActionだけを実行する。
- `commit_action`はrootを進行させる。
- `emulate_action`は既存Branchから新しいBranchを作成し、ActionをSimulationする。
- Branchの状態をrootへ移植しない。採用Actionはroot上で再実行する。
- Trainingから送信されたEmulator DTOを状態復元に使用しない。
- Worker、Snapshot、Replay、Lease、具体的なRNG状態はRL内部に限定する。
- RLは公開可能なEmulator DTOを返し、順序情報や未来情報だけをマスクする。

---

## 2. DTOの項目

### 2.1 共通Request

| 項目 | 型 | 必須条件 |
|---|---|---|
| `schema_version` | string | 常に必須 |
| `request_id` | string | 常に必須 |
| `operation` | string | 常に必須 |
| `instance_id` | string | `start_instance`以外で必須 |

### 2.2 Operation別Request

#### `start_instance`

| 項目 | 必須 |
|---|---|
| 共通Request項目 | `instance_id`を除き必須 |
| `instance_config` | Yes |

#### `get_decision`

| 項目 | 必須 |
|---|---|
| 共通Request項目 | Yes |
| `branch_id` | Yes |

#### `commit_action`

| 項目 | 必須 |
|---|---|
| 共通Request項目 | Yes |
| `branch_id` | Yes。常に`"root"` |
| `rng_id` | Yes。常に`0` |
| `decision_point_id` | Yes |
| `action_id` | Yes |

#### `emulate_action`

| 項目 | 必須 |
|---|---|
| 共通Request項目 | Yes |
| `parent_branch_id` | Yes |
| `branch_id` | Yes |
| `rng_id` | Yes |
| `decision_point_id` | Yes |
| `action_id` | Yes |
| `simulation_options` | No |

#### `cancel_branches`／`release_branches`／`get_branch_status`

| 項目 | 必須 |
|---|---|
| 共通Request項目 | Yes |
| `branch_ids` | Yes |

#### `close_instance`

共通Request項目だけを使用する。

### 2.3 共通Response

常に返す項目：

| 項目 | 型 |
|---|---|
| `schema_version` | string |
| `request_id` | string |
| `operation` | string |
| `status` | string |

状態が存在する場合に返す項目：

| 項目 | 型 |
|---|---|
| `instance_id` | string |
| `branch_id` | string |
| `parent_branch_id` | string |
| `rng_id` | integer |
| `decision_point_id` | string |
| `action_id` | string |
| `branch_log` | array |
| `masked_emulator_dto` | object |

失敗時に返す項目：

| 項目 | 型 |
|---|---|
| `error` | string |
| `fault_kind` | string |

### 2.4 Operation

| Operation | 用途 |
|---|---|
| `start_instance` | Whole Runまたは独立Combatを開始する |
| `get_decision` | 現在のDecisionを取得する |
| `commit_action` | rootへActionを適用する |
| `emulate_action` | 新しいBranchを作成し、ActionをSimulationする |
| `cancel_branches` | Branchを取り消す |
| `release_branches` | Branchの状態と結果を解放する |
| `get_branch_status` | Branchの状態を取得する |
| `close_instance` | instance全体を終了する |

### 2.5 Status

| Status | 意味 |
|---|---|
| `completed` | 正常完了 |
| `partial` | 制限到達時点の有効な途中状態を返した |
| `queued` | Worker割当前 |
| `running` | 実行中 |
| `cancelled` | 取消済み |
| `rejected` | 実行開始前に要求を拒否した |
| `faulted` | 実行開始後に失敗した |
| `released` | Branchを解放済みで再利用できない |

独立した`Error` Statusは設けない。

---

## 3. 各項目の説明

### `schema_version`

RL–Training通信契約のVersion。

TrainingとRLは、対応していないVersionの要求を`rejected`とする。

### `instance_id`

1つの探索木全体を識別するID。

- `start_instance`成功時にRLが発行する。
- rootと全Branchで共有する。
- Whole Runと独立Combatでは別IDを使用する。
- `close_instance`後は再利用しない。

### `request_id`

Trainingが要求ごとに発行するID。

- RLは応答へ同じ値を返す。
- instance内で一意とする。
- 同一要求の再送で処理を二重実行しない。
- 同一IDで内容が異なる場合は`rejected`とする。

### `operation`

RLへ要求する処理の種類。

値は2.4節で定義したOperationに限定する。

### `branch_id`

探索木内の論理Branchを識別するID。

- rootは常に`"root"`。
- 非root BranchはTrainingが発行する。
- instance内で生涯一意とする。
- CancelまたはRelease後も再利用しない。
- Worker ID、PID、generation、Lease IDを含めない。

### `parent_branch_id`

`emulate_action`で、新しいBranchをどの既存Branchから作るかを指定する。

新しい`branch_id`は子Branch自身を識別するが、作成元までは示さないため、親を別項目で指定する。

- `"root"`または既存の非root Branchを指定できる。
- 親Branchは`emulate_action`によって変更されない。
- 親が存在しない、Cancel済み、Release済みの場合は`rejected`とする。

### `decision_point_id`

Branchの現在の選択待ち状態を識別する、RL発行の不透明なID。

- Action適用後、新しいDecisionへ到達するたび更新する。
- Branchごとに管理する。
- 古いIDによる要求は`rejected`とする。
- 別の`state_token`は設けない。

### `action_id`

現在の`masked_emulator_dto.legal_actions`に含まれるAction ID。

- 現在の`decision_point_id`内でのみ有効。
- 別Decisionへ持ち越さない。
- TrainingはAction内容を再構築せず、IDをそのまま返す。

### `rng_id`

Trainingから見える論理的なRNG Hypothesis ID。

- rootは常に`0`。
- Simulation Branchでは正の整数を使用する。
- seed、RNG内部状態、DrawPile順序は公開しない。
- 同じ親Decisionで同じ`rng_id`を使った要求は、同じHypothesisを共有する。
- 異なる`rng_id`は異なるHypothesisを示す。
- `emulate_action`で`rng_id=0`は使用できない。
- 非root Branchからさらに分岐する場合、v0.5では親Branchと同じ`rng_id`を使用する。

Hypothesisの識別単位：

`(instance_id, parent_branch_id, decision_point_id, rng_id)`

### `branch_log`

rootから現在のBranchまでの論理Action履歴。

- root開始時は空配列。
- RLがAction成功後に追加する。
- 子Branchは親のLogを引き継ぐ。
- Trainingは通常のRequestで返送しない。
- Trajectory、debug、経路確認に使用する。
- Branch状態の正本や親Branch指定には使用しない。

各要素は以下を持つ。

- `depth`
- `decision_point_id`
- `action_id`
- `rng_id`

### `simulation_options`

`emulate_action`の停止条件と実行上限。

| 項目 | 意味 |
|---|---|
| `stop_condition` | `next_decision`、`combat_end`、`room_end`、`run_end` |
| `max_depth` | 追加Decision深度の上限 |
| `max_steps` | Emulator Step数の上限 |
| `max_time_ms` | 実行時間の上限 |
| `max_hypotheses` | Hypothesis数の安全上限 |

未対応の条件は`rejected`とする。

制限到達時点の有効な状態を返せる場合は`partial`とする。

### `status`

要求またはBranchの状態。

- `rejected`は実行開始前の拒否であり、元状態を変更しない。
- `faulted`は実行開始後の失敗を示す。
- `released`のBranchは状態と結果を再利用できない。

### `fault_kind`

Trainingが再試行判断に使用する機械向けの障害分類。

想定値：

- `task_timeout`
- `worker_process_crash`
- `replay_mismatch`
- `emulator_error`
- `snapshot_restore_failed`

### `error`

人間向けのエラー文字列。

- `rejected`または`faulted`の場合だけ設定する。
- Trainingはこの文字列を機械的に解析しない。

### `masked_emulator_dto`

Emulator DTOからHidden InformationとRL内部情報を除去したDTO。

`dto_version`と`mask_version`を含める。

#### 削除する情報

- Snapshot／SaveState
- RNG内部状態
- Worker ID／PID／generation
- Lease
- Replay Prefix
- 内部Context／Snapshot識別子
- CombatSessionId
- allowlist外の`Metrics`
- allowlist外の`Extras`
- allowlist外の`Info`
- Emulator固有のCombat Reward
- 将来のEvent／Encounter列
- 将来のBoss／Ancient情報
- 事前生成Queueのcursorと順序

#### マスクする情報

- `drawPile`：順序を除去してMultiset化
- `discardPile`：順序を除去してMultiset化
- `exhaustPile`：順序を除去してMultiset化
- `playPile`：mask version 1.0では削除
- `Transition.FinalObservation`：同じ規則を再適用

#### 過去のEvent／Encounter履歴

実際にrootまたはBranch上で発生した履歴だけを公開する。

公開可能：

- 訪問済みEvent ID
- 完了済みEncounter ID
- 過去のElite／Boss
- 選択済みEvent Option
- 訪問済みMap座標

非公開：

- 未到達Event／Encounter列
- 将来Queue順序
- Hidden Queueのcursor
- 将来Boss／Ancient情報

履歴は実際のTransitionから逐次構築し、事前生成列から作成しない。

#### そのまま公開する情報

- Legal Actions
- HP／Max HP
- Gold
- Energy／Block
- Hand
- 永続Deck
- Relics
- Potions
- Orbs
- Powers
- Enemy公開状態とIntent
- 公開済みMap Room種別
- 現在のEvent／Reward／Shop／Rest選択肢
- Boundary
- Room Context
- Transition Outcomeの公開部分

---

## 4. 補足

### 4.1 rootとBranch

#### root

- instanceに1つだけ存在する。
- `branch_id="root"`。
- `rng_id=0`。
- `commit_action`だけで進行する。
- `emulate_action`では変化しない。

#### Branch

- `emulate_action`で作成する。
- 親Branchを1つ持つ。
- Training発行の一意な`branch_id`を持つ。
- 正の`rng_id`を持つ。
- さらに深いBranchの親になれる。
- rootや親Branchを上書きしない。
- Cancel／Releaseできる。

### 4.2 root Commit後のBranch

`commit_action`成功後、直前のroot Decisionから派生したBranchはstaleとなる。

RLは次を行う。

1. root上でActionを再実行する。
2. rootを次Decisionまで進める。
3. 対象BranchをCancelする。
4. 関連Leaseを無効化する。
5. BranchをReleaseする。

Simulation結果をrootへ移植しない。

### 4.3 Branch管理

- `cancel_branches`は冪等とする。
- running BranchのCancelではWorker killとRespawnを許可する。
- Holder BranchのCancelではLeaseを無効化する。
- `release_branches`は必要なら先にCancelする。
- Release後は状態と結果を再利用できない。
- Branch IDはRelease後も再利用できない。
- `close_instance`は全BranchとLeaseを解放する。

### 4.4 Request／Response例

#### `start_instance`

```json
{
  "schema_version": "0.5",
  "request_id": "req-001",
  "operation": "start_instance",
  "instance_config": {
    "instance_type": "whole_run",
    "character_id": "IRONCLAD",
    "ascension": 10,
    "seed": 18
  }
}
```

#### `commit_action`

```json
{
  "schema_version": "0.5",
  "request_id": "req-003",
  "operation": "commit_action",
  "instance_id": "inst-001",
  "branch_id": "root",
  "rng_id": 0,
  "decision_point_id": "d-root-001",
  "action_id": "a-003"
}
```

#### `emulate_action`

```json
{
  "schema_version": "0.5",
  "request_id": "req-004",
  "operation": "emulate_action",
  "instance_id": "inst-001",
  "parent_branch_id": "root",
  "branch_id": "branch-001",
  "rng_id": 1,
  "decision_point_id": "d-root-001",
  "action_id": "a-003",
  "simulation_options": {
    "stop_condition": "next_decision",
    "max_depth": 1,
    "max_steps": 100,
    "max_time_ms": 5000
  }
}
```

#### Response

```json
{
  "schema_version": "0.5",
  "instance_id": "inst-001",
  "request_id": "req-004",
  "operation": "emulate_action",
  "status": "completed",
  "branch_id": "branch-001",
  "parent_branch_id": "root",
  "rng_id": 1,
  "decision_point_id": "d-branch-001-002",
  "action_id": "a-003",
  "branch_log": [
    {
      "depth": 0,
      "decision_point_id": "d-root-001",
      "action_id": "a-003",
      "rng_id": 1
    }
  ],
  "masked_emulator_dto": {
    "dto_version": "emulator-fca2f06",
    "mask_version": "1.0"
  },
  "fault_kind": null,
  "error": null
}
```

### 4.5 RLが拒否する要求

- staleな`decision_point_id`
- Legalでない`action_id`
- 使用済み`branch_id`
- 内容が競合する重複`request_id`
- `emulate_action`での`rng_id=0`
- 非root Branchで親と異なる`rng_id`
- Cancel／Release済みBranchへの実行要求
- rootへのCancel／Release
- TrainingのDTOを状態復元に使用する要求

### 4.6 RLが保証する事項

- `rejected`要求では状態を変更しない。
- Branch Simulationでrootと親Branchを変更しない。
- Worker Faultでrootを変更しない。
- 古いWorker generationの結果を採用しない。
- Cancel／Release時に必要なLeaseを無効化する。
- Hidden Informationを`masked_emulator_dto`へ含めない。
