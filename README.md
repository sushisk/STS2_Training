# STS2_Training API connection

Training-side implementation for the `sushisk/STS2_RL` API v0.5.

## Asyncio TCP smoke test

RLとTrainingを別プロセスとして起動し、UTF-8 newline-delimited JSONで疎通します。
まず`STS2_RL`側を起動します。

```bash
python -m API.tcp_server --host 127.0.0.1 --port 8765
```

別のプロセスでTraining側のpingを実行します。

```bash
python -m sts2_training.api.tcp_smoke --host 127.0.0.1 --port 8765
```

成功時は `{"transport_operation": "pong"}` が表示されます。

## Async DTO API client

TCP経路では、APIの意味とTCP接続の責務を分離します。

- `ApiContract`: DTO生成、request/response相関、instance追跡、validation、selection audit
- `AsyncTrainingApiClient`: API v0.5操作をasyncで実行し、DTOの生成・検証を制御
- `TcpConnection`: connect / NDJSON encode-decode / timeout / reconnect のみ
- `RLApiServer`: `operation` と `instance_id` による実処理の振り分け

`AsyncTrainingApiClient`は「何を送るか・返答が正しいか」を担当し、`TcpConnection`は
「JSON objectをどうTCPで1往復させるか」だけを担当します。`TcpConnection`はAPI operationや
Instanceを解釈しません。

v0.5では1接続上のrequest/responseを直列化するため、TCP専用のinternal IDやresponse routerは
持ちません。相関はDTO自身の`request_id`を`AsyncTrainingApiClient`が検証します。

デフォルトの`request_id`はUUIDベースです。Training processを再起動したりclientを作り直したりしても
`req-000001`のような短いcounterへ戻らないため、RL server-wideのrequest ledgerと衝突しません。

```python
import asyncio

from sts2_training.api import AsyncTrainingApiClient, TcpConnection


async def main() -> None:
    connection = TcpConnection(host="127.0.0.1", port=8765)
    async with AsyncTrainingApiClient(connection) as client:
        instance_id = await client.start_instance(
            {"instance_type": "combat"},
            timeout_s=30.0,
        )
        decision = await client.get_decision(instance_id, timeout_s=30.0)
        print(decision)


asyncio.run(main())
```

同期版`TrainingApiClient`は既存の`RlTransport` / `LocalProcessTransport`用として残していますが、
非同期TCP Clientとは継承関係にありません。両Clientは通信非依存の`ApiContract`だけを共有します。

TCP上ではDTO自体をUTF-8 newline-delimited JSONの1フレームとして送ります。
`schema_version` / `request_id` / `operation` / `instance_id` の相関規則はAPI v0.5の
DTO契約と同一です。

`TcpConnection(max_message_bytes=...)` の上限は**送信request frameだけ**に適用されます。
API responseには同じ上限を適用せず、newlineまで1フレームを読み切ります。これは
`commit_action` / `emulate_action`のようなnon-idempotent operationが成功した後に、大きいresponseを
transport層で捨ててsame-ID retryでも永久に結果を回収できなくなる事態を避けるためです。

### Timeout / cancellation / retry semantics

TCP timeoutは「RLが未実行」という意味ではなく、「Trainingが結果を観測できなかった」という
ambiguous completionです。送信開始後にtimeoutまたはtask cancellationが起きた場合、
`TcpConnection`はそのstreamを破棄します。request_id/operation mismatchなどAPI correlation failureを
検出した場合も、次のcallの前にconnectionをinvalidateします。

同一の論理requestをretryする場合は、**同じpayloadと同じ`request_id`**をfresh connection上で再送します。
新しい`request_id`で`start_instance` / `commit_action` / `emulate_action`を呼び直すと別の論理requestになり、
二重実行の可能性があります。raw DTOを保持しているrecovery codeでは`TcpConnection.exchange()`へ
同じdictを再度渡すことでsame-ID retryできます。高レベルAPIはambiguous completionを自動的に
「成功」または「未実行」と推測しません。

`ApiContract`のactive instance / selection audit stateは、correlated API responseを正常に受理した時だけ
更新されます。したがってtimeoutした`start_instance`ではローカルstateは未開始のまま、timeoutした
`close_instance`ではローカルstateはactiveのままです。これはRL stateが確定したことを意味しないため、
同一requestのretry/replayで結果をreconcileしてから次の論理操作へ進みます。

transport-only response（例: oversized requestに対する
`{"transport_error":"message_too_large","direction":"request", ...}`）はAPI DTOではなく
`TransportError`として扱われ、API envelope validationには渡しません。

## Unit tests

```bash
python -m pytest tests/api -m "not integration"
```

### Cross-repository TCP contract tests

`STS2_RL` checkoutを`PYTHONPATH`へ追加すると、Training側からreal
`AsyncioTcpServer + RLApiServer`へ接続するcontract testも実行できます。RL checkoutが見つからない通常環境では
このmoduleだけskipされます。

```bash
PYTHONPATH=src:../STS2_RL python -m pytest tests/api/test_rl_cross_repo_contract.py -vv
```

このtestは、client再生成後の`start_instance` request_id collision、fresh connection上のsame-ID replay、
responseを観測できなかった`close_instance`のretry、およびrequest frame limitを超えるサイズの成功responseを
返すnon-idempotent `commit_action`が実responseを受信・same-ID replayでき、処理自体は1回しか実行されないことを
確認します。

## Selection audit logging

Pass a `JsonlSelectionLogger` to `TrainingApiClient` or `AsyncTrainingApiClient` to append
one flushed UTF-8 JSON record for each `commit_action` or `emulate_action` call.

```python
from sts2_training.api.client import TrainingApiClient
from sts2_training.selection_log import JsonlSelectionLogger

with JsonlSelectionLogger("logs/selection.jsonl") as selection_log:
    client = TrainingApiClient(
        transport,
        selection_logger=selection_log,
    )
```

Each record contains the public Decision received from RL, the selection request, and the
correlated result. A successful root selection also includes `room_result` when it ends a
room and `run_result` when it ends a Whole Run. Rejected and faulted selections are logged
before their exception is raised.

Only the already-masked DTO received from RL is written. Training does not reconstruct or
add hidden state, and speculative Branch results are never counted as root room/run
results.

## Real Emulator integration test (Windows cmd.exe)

```bat
set STS2_RL_ROOT=C:\path\to\STS2_RL
python -m pytest tests/api/test_api_smoke.py -m integration -vv
```

The combat smoke test covers the complete runtime path without a separate connection
test: Training starts the spawned RL process, `start_instance` loads CoreCLR and the
Emulator and returns a real decision, a Branch Worker executes one speculative action,
and the root action is committed and closed.

## Random root progression integration tests

```bat
set STS2_RL_ROOT=C:\path\to\STS2_RL
python -m pytest tests/api/test_api_random_progression.py -m integration -vv
```

These tests start a fixed independent Combat and a fixed Whole Run, never call
`emulate_action`, and select directly from each published root `legal_actions` list with
a locally seeded PRNG. They verify the configured Combat state, renewed decision IDs,
contiguous root branch logs, at least one non-first random choice, and several distinct
public board states after removing `legal_actions` and DTO-version metadata from the
progress fingerprint.

`LocalProcessTransport` imports `API.api_runtime.RLApiServerProcess` from
`STS2_RL_ROOT`. CLR initialization remains inside the spawned RL child process.
