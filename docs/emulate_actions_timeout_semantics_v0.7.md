# `emulate_actions` timeout semantics (v0.7)

`emulate_actions` has two independent clocks in the paired RL/Training API. They must not
be configured as though they were the same deadline.

## RL: `simulation_options.max_time_ms`

For Combat, `simulation_options.max_time_ms` is a **per-Branch execution timeout**. The
clock for a Branch starts only when `BranchManager` dispatches that Branch to a worker.
Branches that are still coordinator-side `queued` do not consume this timeout. With
`items > worker_count`, later items therefore start their own fresh timeout only after a
worker becomes available.

This means `max_time_ms` is **not** a wall-clock deadline for the complete batch. For
example, with two workers and four items, a healthy batch can require roughly two waves
of Branch execution and legitimately take longer than one `max_time_ms` interval.

## Training: `timeout_s`

`AsyncTrainingApiClient.emulate_actions(..., timeout_s=...)` uses `timeout_s` to compute
one absolute client I/O deadline. That deadline covers waiting for the serialized
operation slot and the TCP exchange through receipt of the complete response frame. It
is independent of RL's per-Branch execution timer.

`timeout_s` is **not** a strict wall-clock deadline for synchronous local processing that
happens after the complete response frame has been received. JSON decoding, server-epoch
checks, DTO correlation/validation, and SelectionAudit bookkeeping may finish after that
absolute I/O deadline without converting an otherwise valid response into a timeout.
This distinction is intentional: once the full response frame has arrived, treating later
local CPU work as an uncertain transport timeout would unnecessarily enter exact-replay
recovery even though the response bytes are already available locally.

Therefore callers must **not** assume that
`timeout_s == simulation_options.max_time_ms / 1000` is sufficient for a batch. A batch
larger than the worker count can be healthy on RL while Training reaches its shorter
client I/O deadline first.

If the client I/O deadline expires after the request may have been sent, completion is
uncertain. Training keeps the exact serialized request as `pending_retry`; the caller
must replay that exact request through `retry_request()` before sending a fresh request.
RL's session replay cache then provides the existing at-most-once behavior for the whole
batch. Conversely, if the complete response frame arrives before the deadline, later
local validation may return after `timeout_s` and is still treated as a normal response
when validation succeeds.

## Caller guidance

Choose `max_time_ms` from the maximum acceptable execution time of one dispatched Branch.
Choose `timeout_s` independently to cover serialized-slot wait, queueing across worker
waves, transport overhead, and receipt of the complete response frame. Do not treat
`timeout_s` as a CPU-processing budget for post-frame JSON/DTO/audit work. Beam or other
wide-frontier callers should chunk to `max_emulate_actions_items` and should still budget
`timeout_s` for the number of worker waves in each chunk.

## Shared result queue ownership

Under v0.7's single in-flight/global-handler-lock model, a result whose `request_id` is not
owned by the current `BranchManager.poll()` can only be a stale/late result from a request
that was already cancelled or timed out. `poll()` discards such a stale result and keeps
waiting for the result owned by the current poll. A regression test fixes this behavior.
Concurrent independent polls remain outside the v0.7 contract; removing the global
serialization would require an explicit result-routing/ownership design rather than
relying on stale-result discard.
