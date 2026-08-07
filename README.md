# STS2_Training API connection

Training-side implementation for the `sushisk/STS2_RL` async TCP / DTO v0.6 contract.

The supported path is now deliberately **async + TCP only**. The legacy synchronous
`TrainingApiClient` / `LocalProcessTransport` path is retired for v0.6 because it cannot
participate in the session handshake and `server_epoch` safety model.

## TCP smoke test

Start RL separately, then run:

```bash
python -m sts2_training.api.tcp_smoke --host 127.0.0.1 --port 8765
```

A new TCP stream first sends an exact transport hello containing a stable
`client_session_id`. RL returns its `server_epoch`. The smoke test then sends ping and
prints a response similar to:

```json
{"server_epoch":"...","transport_operation":"pong"}
```

## Async DTO API client

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

## Session sequencing

Each client owns one `client_session_id` and sends strictly increasing `request_seq`
values. `request_id` is deterministic: `<client_session_id>:<request_seq>`.

The client advances its sequence only after receiving a definitive API response. If a
request may have reached RL but no valid response was observed, the exact serialized DTO
is exposed as `client.pending_retry` and all fresh operations fail closed.

Recovery is explicit:

```python
pending = client.pending_retry
if pending is not None:
    result = await client.retry_request(pending, timeout_s=30.0)
```

RL keeps the most recent executable request/response for every logical session. Exact
same-sequence retry is therefore replayed rather than executed again. A different payload
with the same sequence or a sequence gap is rejected.

## RL restart semantics

Every API response and transport hello/pong contains `server_epoch`. A reconnect must see
the same epoch. If RL restarted, `TcpConnection` raises `ServerEpochChangedError` and the
`AsyncTrainingApiClient` becomes permanently invalid. Create a new client/session; do not
retry the unresolved request into the new RL process.

This is intentional: v0.6 guarantees at-most-once execution within one RL process epoch,
not durable exactly-once execution across emulator process restarts.

## Timeouts, cancellation, and response limits

A public API `timeout_s` covers waiting for the client operation lock and the TCP
exchange. Cancellation/timeout after send invalidates the stream and preserves the exact
pending request.

`TcpConnection(max_response_bytes=...)` bounds response buffering independently from the
request frame limit. If a valid cached response is larger than the local receiver bound,
raise the bound with `await connection.set_max_response_bytes(...)` and replay the exact
`pending_retry`; changing the local receiver limit does not change request identity.

## Selection audit

`AsyncTrainingApiClient` still accepts a `SelectionEventLogger`. A completion-uncertain
selection is logged on the first attempt; replay of the same `request_id` is recorded as
selection recovery rather than a second logical selection.
