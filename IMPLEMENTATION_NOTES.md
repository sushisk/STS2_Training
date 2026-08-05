# 実装済み範囲と学習停止点

## 実装済み

- `RlTransport` Protocol / `FakeTransport`
- `TrainingApiClient` の8 Operation
  - `start_instance`
  - `get_decision`
  - `commit_action`
  - `emulate_action`
  - `cancel_branches`
  - `release_branches`
  - `get_branch_status`
  - `close_instance`
- Request ID発行、共通Response envelope照合
- Operation固有Requestのcross-field制約
- `rejected` / `faulted`を通信不整合と区別
- 全OperationのUnit Test

## 意図的に未実装

- `LocalProcessTransport`
- `RLApiServerProcess` のspawnと終了
- timeout後の遅延Response demultiplex
- retry（同じrequest_idと同じpayloadを再利用する必要がある）
- Pydantic DTO
- `EpisodeSession` / `EpisodeController`

## 次に理解すべき境界

`LocalProcessTransport`は単なるRequest生成ではなく、子プロセスの所有者になります。
ここでは次を理解してから実装します。

1. 親プロセスと子プロセスのメモリは共有されない
2. `spawn`では子側がmoduleを再importする
3. Queueへ渡せるのはpickle可能なJSON-safe値
4. timeoutしても子側の処理は終了していない可能性がある
5. 遅れて到着したResponseを次RequestのResponseとして誤受理してはいけない
6. `close_instance()`と`transport.close()`は別のライフサイクル
