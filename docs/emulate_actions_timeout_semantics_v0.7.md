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

`AsyncTrainingApiClient.emulate_actions(..., timeout_s=...)` uses `timeout_s` as the
client-side deadline for the whole API operation/exchange. It includes time spent waiting
for the serialized operation slot and waiting for the batch response. It is independent
of RL's per-Branch execution timer.

Therefore callers must **not** assume that
`timeout_s == simulation_options.max_time_ms / 1000` is sufficient for a batch. A batch
larger than the worker count can be healthy on RL while Training reaches its shorter
client deadline first.

If the client deadline expires after the request may have been sent, completion is
uncertain. Training keeps the exact serialized request as `pending_retry`; the caller
must replay that exact request through `retry_request()` before sending a fresh request.
RL's session replay cache then provides the existing at-most-once behavior for the whole
batch.

## Caller guidance

Choose `max_time_ms` from the maximum acceptable execution time of one dispatched Branch.
Choose `timeout_s` independently from the acceptable end-to-end batch latency, including
queueing across worker waves, transport overhead, and response processing. Beam or other
wide-frontier callers should chunk to `max_emulate_actions_items` and should still budget
`timeout_s` for the number of worker waves in each chunk.
