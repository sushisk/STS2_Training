# api モジュール

## 0. 文章の目的

この文書は `src/sts2_training/api/` が担う Training API 接続層を説明する。対象は TCP/async client、DTO contract validation、response routing、local process transport、smoke test であり、wire protocol 詳細そのものは [STS2_wire_contract_v0.8.md](STS2_wire_contract_v0.8.md) を参照する。

## 1. 概要

`api/` は `STS2_RL` の Training API と `STS2_Training` の意思決定コードをつなぐ境界である。`AsyncTrainingApiClient` は logical request stream を 1 本の asyncio TCP connection に直列化し、`ApiContract` が DTO v0.8 の request construction、response validation、active instance state、selection audit を管理する。

現在の contract constants は `SCHEMA_VERSION = "0.8"`、`MASK_VERSION = "1.2"`、root branch は `ROOT_BRANCH_ID = "root"`、root RNG は `ROOT_RNG_ID = 0` である。TCP layer は hello/ping、server epoch の固定、NDJSON frame size limit、timeout 後の completion uncertainty と `RetryRequest` を扱う。

## 2. Architecture

主要ファイルは次の役割を持つ。

| ファイル | 役割 |
|---|---|
| `contract.py` | operation DTO を組み立て、response の schema/status/correlation/branch result を検証する |
| `async_client.py` | `ApiContract` を継承し、request sequence、operation lock、retry_request、timeout を持つ async client |
| `tcp_connection.py` | UTF-8 NDJSON TCP connection、hello handshake、epoch validation、message size limit |
| `transport.py` | `RetryRequest` と transport exception 群 |
| `response_router.py` | private transport internal ID ごとの async response routing |
| `local_process_transport.py` | sibling `STS2_RL` の `RLApiServerProcess` を同一 process から呼ぶ transport |
| `client.py` | legacy synchronous compatibility surface |
| `tcp_smoke.py` | 外部起動済み TCP server への最小疎通確認 CLI |

`ApiContract` は `start_instance` を accept すると active `instance_id` と `instance_type` を保持し、`close_instance` でクリアする。`emulate_actions` は batch item の `branch_id` 重複と「同じ batch 内で作った branch を parent にする」形をローカルで拒否する。`commit_action` は root branch/root rng に固定される。

`TcpConnection` は server epoch が途中で変わると `ServerEpochChangedError` を投げる。request bytes が送信済みの timeout/connection loss は completion が不確実なため、exception に `retry_request` を付け、`AsyncTrainingApiClient.retry_request()` で同じ request を再送できる設計になっている。

## 3. API

主要 public interface は次である。

```python
class AsyncTrainingApiClient(ApiContract):
    def __init__(self, transport: RlTransport | None = None, *, host="127.0.0.1", port=8765, ...)
    async def start_instance(self, instance_config: Mapping[str, Any], *, timeout_s: float = ...) -> str
    async def get_decision(self, instance_id: str, branch_id: str = "root", *, timeout_s: float = ...) -> dict[str, Any]
    async def commit_action(self, instance_id: str, decision_point_id: str, action_id: str, *, timeout_s: float = ...) -> dict[str, Any]
    async def emulate_action(...)
    async def emulate_actions(...)
    async def cancel_branches(...)
    async def release_branches(...)
    async def get_branch_status(...)
    async def close_instance(...)
    async def retry_request(retry_request: RetryRequest, *, timeout_s: float = ...) -> dict[str, Any]
```

```python
class TcpConnection:
    async def connect(self) -> None
    async def exchange(self, message: Mapping[str, Any], *, timeout_s: float | None = None, deadline: float | None = None) -> dict[str, Any]
    async def ping(self, *, timeout_s: float = 5.0) -> dict[str, Any]
    async def close(self) -> None
```

```python
@dataclass(frozen=True)
class RetryRequest:
    serialized_payload: str
    @classmethod
    def from_message(cls, message: Mapping[str, Any]) -> "RetryRequest"
    def to_message(self) -> dict[str, Any]
```

## 4. 使用例

```python
import asyncio

from sts2_training.api import AsyncTrainingApiClient


async def main() -> None:
    async with AsyncTrainingApiClient(host="127.0.0.1", port=8765) as client:
        instance_id = await client.start_instance({
            "instance_type": "whole_run",
            "seed": 123,
        })
        decision = await client.get_decision(instance_id, "root")
        action = decision["masked_emulator_dto"]["legal_actions"][0]
        await client.commit_action(
            instance_id,
            decision["decision_point_id"],
            action["action_id"],
        )
        await client.close_instance(instance_id)


asyncio.run(main())
```

TCP server の疎通だけを確認する場合:

```bash
python -m sts2_training.api.tcp_smoke --host 127.0.0.1 --port 8765 --timeout 5
```

## 5. 補足説明

`client.py` は legacy 同期 surface であり、新しい実行経路は `AsyncTrainingApiClient` を使う。`local_process_transport.py` は sibling repo の import origin を検証し、別 package が混ざる事故を避ける。Combat search 側の使い方は [02_decision_core.md](02_decision_core.md)、runner CLI からの接続は [07_runner_cli.md](07_runner_cli.md) を参照する。
