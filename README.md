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

TCP経路は責務を3つに分けます。

- `ApiContract`: DTO生成・validation・correlation・client state
- `AsyncTrainingApiClient`: async API operationの実行制御
- `TcpConnection`: connect / NDJSON / timeout / reconnect

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

`AsyncTrainingApiClient`は現在、client-level lockでpublic operationを直列化します。
デフォルト`request_id`はUUIDベースです。

各public APIの`timeout_s`はmethodを呼んだ時点からresponse frame受信までのtransport budgetです。
client-level lock待ち、`TcpConnection`のlock待ち、connect、request write/drain、response readは
同じabsolute deadlineを消費します。response受信後のAPI validationとselection audit bookkeepingは
transport timeout phaseには含めません。`connect_timeout_s`はconnect phaseだけの追加上限であり、
APIのdeadlineを延長しません。

requestがRLに到達した可能性がある状態でstate-changing operationの結果を確定できなかった場合、
`TransportError.completion_uncertain=True`となり、exact serialized requestを保持する
`RetryRequest`が`TransportError.retry_request`と`client.pending_retry`から取得できます。
この状態ではfresh requestをfail-closedで拒否します。同じ論理requestを再送する場合は
`await client.retry_request(client.pending_retry, timeout_s=...)`を使い、same payload / same
`request_id`を維持してください。

`start_instance`の不確実性は`start_uncertain`、`close_instance`の不確実性は`close_uncertain`でも
確認できます。same-ID replayで回復できない場合、外部確認やoperator判断の後に
`reconcile_start_uncertainty(instance_id=...)`または
`reconcile_close_uncertainty(assume_closed=True|False)`でlocal stateを明示的に解消できます。
これらのreconciliation method自体はRLへ通信しません。自動retry、RL state discovery、
uncertain instanceの自動cleanupはこの実装では定義しません。

`TcpConnection(max_message_bytes=...)` の上限は送信request frameにのみ適用します。responseは
別の `max_response_bytes`（default 64 MiB）でbufferingをboundedにします。response上限超過は
request送信後に起こり得るため`completion_uncertain=True`の`TransportError`となり、connectionを
破棄します。transport-only errorはAPI DTO validationの前に`TransportError`として扱います。

同期版`TrainingApiClient`は既存の`RlTransport` / `LocalProcessTransport`用として残し、
両clientは`ApiContract`を共有します。

## Unit tests

```bash
python -m pytest tests/api -m "not integration"
```

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
before their exception is raised. Async selections cancelled after dispatch are also
recorded with `client_error.type == "CancelledError"` before cancellation is re-raised.

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
