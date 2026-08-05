# RLApiServer / server.py Usage

`C:\STS2_RL\TrainingAPI\server.py` は、Training と RL Runtime の間の要求を受け取り、適切な `Instance` に振り分けて共通の Response 形式に整える dispatcher です。

この README は、`server.py` と `C:\STS2_Training\rl_training_dto_documentation_v0_5.md` の内容を元に、使い方を実務向けに整理したものです。

## 役割

- `start_instance` で `CombatInstance` または `WholeRunInstance` を生成する
- `get_decision` / `commit_action` / `emulate_action` / `cancel_branches` / `release_branches` / `get_branch_status` を各 instance に委譲する
- `close_instance` で instance を終了し、管理対象から外す
- すべての Response を `schema_version`, `request_id`, `operation` を含む共通形式に包む
- 同一 request の再送に対しては ledger により重複処理を避ける

## 前提

- Request と Response は JSON-safe な `dict` でやり取りする
- `schema_version` は `0.5`
- `operation` は DTO で定義された値に限定する
- `start_instance` 以外の操作では `instance_id` が必要

## 起動構成

`server.py` 自体は単独の transport ではなく、`TrainingAPI.api_runtime.RLApiServerProcess` から使われます。

- `RLApiServerProcess` が spawn された子プロセス内で `RLApiServer()` を生成する
- 親プロセスは `call(payload)` で request を送る
- サーバ側は request を受けて `handle_request(payload)` を実行する

## 対応 Operation

`server.py` が直接扱う operation は次の 8 種類です。

- `start_instance`
- `get_decision`
- `commit_action`
- `emulate_action`
- `cancel_branches`
- `release_branches`
- `get_branch_status`
- `close_instance`

## Request 形式

### 共通フィールド

`start_instance` 以外では次の共通項目が必要です。

- `schema_version`
- `request_id`
- `operation`
- `instance_id`

### `start_instance`

必須:

- `schema_version`
- `request_id`
- `operation`
- `instance_config`

`instance_id` は不要です。成功時に RL 側が新規発行します。

### `get_decision`

必須:

- 共通項目
- `branch_id`

### `commit_action`

必須:

- 共通項目
- `branch_id` は常に `root`
- `rng_id` は常に `0`
- `decision_point_id`
- `action_id`

### `emulate_action`

必須:

- 共通項目
- `parent_branch_id`
- `branch_id`
- `rng_id`
- `decision_point_id`
- `action_id`

任意:

- `simulation_options`

### `cancel_branches` / `release_branches` / `get_branch_status`

必須:

- 共通項目
- `branch_ids`

### `close_instance`

必須:

- 共通項目のみ

## Response 形式

### 正常系

Response には少なくとも次が入ります。

- `schema_version`
- `request_id`
- `operation`
- `status`

操作によって追加で返ることがあります。

- `instance_id`
- `branch_id`
- `parent_branch_id`
- `rng_id`
- `decision_point_id`
- `action_id`
- `branch_log`
- `masked_emulator_dto`

### 異常系

拒否時は次を返します。

- `schema_version`
- `request_id`
- `operation`
- `status = rejected`
- `error`
- `fault_kind`

## server.py の内部動作

`handle_request(payload)` の流れは次の通りです。

1. `validate_request(payload)` で DTO を検証する
2. 検証失敗なら `status = rejected` で返す
3. `start_instance` なら pre-instance ledger を確認して重複実行を避ける
4. それ以外は `instance_id` から対象 instance を引く
5. instance の ledger を確認して同一 request の再実行を避ける
6. operation に応じて instance メソッドへ委譲する
7. Response を共通ヘッダ付きに包む
8. `close_instance` の場合は instance を close して管理表から削除する

## 実行シーケンス

### 1. `start_instance`

```python
payload = {
    "schema_version": "0.5",
    "request_id": "req-000001",
    "operation": "start_instance",
    "instance_config": {
        "instance_type": "combat",
        "...": "..."
    }
}
```

成功すると `instance_id` が発行されます。以後の request はその `instance_id` を使います。

### 2. `get_decision`

`branch_id` を指定して現在の選択待ち状態を取得します。通常は `root` から開始します。

### 3. `commit_action`

root を進行させる操作です。

- `branch_id` は常に `root`
- `rng_id` は常に `0`
- `decision_point_id` と `action_id` を指定する

成功後、直前の root decision から派生した branch は stale になります。

### 4. `emulate_action`

既存 branch から新しい branch を作って action をシミュレーションします。

- `parent_branch_id` で親 branch を指定する
- `branch_id` は新しい branch の識別子
- `rng_id` は論理的な RNG hypothesis の識別子
- `simulation_options` で停止条件や上限を指定できる

### 5. `cancel_branches` / `release_branches`

複数 branch をまとめて操作します。

- `cancel_branches` は取り消し
- `release_branches` は状態と結果の解放

### 6. `get_branch_status`

指定 branch 群の状態を確認します。

### 7. `close_instance`

instance を終了し、サーバ側の管理から外します。

## 実装上の注意

- `start_instance` の `instance_type` は `combat` または `whole_run`
- `instance_id` が存在しない場合は `unknown instance_id` として拒否される
- `server.py` は `instance_combat.py` と `instance_whole_run.py` の唯一の dispatcher
- `RLApiServerProcess` から使う場合、timeout 時は `status = faulted` と `fault_kind = task_timeout`
- `handle_request` は request ごとに ledger を使い、同一 request の二重実行を防ぐ

## 典型的な使い方

### Training 側

1. `RLApiServerProcess()` を生成する
2. `start_instance(instance_config)` を呼ぶ
3. `get_decision(branch_id="root")` を呼ぶ
4. `legal_actions` から `action_id` を選ぶ
5. `commit_action(...)` または `emulate_action(...)` を呼ぶ
6. 必要に応じて `cancel_branches(...)` / `release_branches(...)` / `close_instance()` を呼ぶ

### テスト用の簡易呼び出し

`TrainingAPI.mock_training_client.MockTrainingClient` を使うと、request_id と branch_id の生成を補助できます。

## 参照

- [`C:\STS2_RL\TrainingAPI\server.py`](C:\STS2_RL\TrainingAPI\server.py)
- [`C:\STS2_Training\rl_training_dto_documentation_v0_5.md`](C:\STS2_Training\rl_training_dto_documentation_v0_5.md)

