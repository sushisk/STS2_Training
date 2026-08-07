# STS2 RL / Training wire contract v0.7

This document is the canonical contract for the `0.7` RL/Training API wire format.
Version `0.6` remains historical; `0.7` adds the plural `emulate_actions` operation while
preserving the existing single in-flight session-sequenced transport model.

## Architecture

The intended execution model remains:

```text
Async I/O
    ↓
Single-threaded Coordinator
    ↓
Parallel Worker Processes
```

`BranchManager` is not made async. Requests are not multiplexed, and Training does not
have multiple API requests in flight at once.

## Common request identity

Every request carries the existing session fields:

```text
schema_version = "0.7"
client_session_id
request_seq
request_id = f"{client_session_id}:{request_seq}"
operation
instance_id
```

The current request must be completed or exact-replayed before a fresh request can be
sent. `request_seq` advances only after the operation-specific response is accepted.

## `emulate_actions` request

`operation` is `"emulate_actions"` and the request contains a non-empty `items` array.
Optional `simulation_options` uses the same validation rules as `emulate_action`.

Conceptual shape:

```json
{
  "schema_version": "0.7",
  "client_session_id": "session-a",
  "request_seq": 42,
  "request_id": "session-a:42",
  "operation": "emulate_actions",
  "instance_id": "inst-001",
  "items": [
    {
      "parent_branch_id": "b1",
      "branch_id": "c1",
      "rng_id": 1,
      "decision_point_id": "d-b1-003",
      "action_id": "a-001"
    },
    {
      "parent_branch_id": "b2",
      "branch_id": "c2",
      "rng_id": 2,
      "decision_point_id": "d-b2-004",
      "action_id": "a-002"
    }
  ],
  "simulation_options": {
    "stop_condition": "next_decision"
  }
}
```

### Batch item schema

Each item contains exactly the branch/action identity needed by `emulate_action`:

- `parent_branch_id`: non-empty string.
- `branch_id`: non-empty, non-`root` string and unique within the batch.
- `rng_id`: positive integer.
- `decision_point_id`: non-empty string.
- `action_id`: non-empty string.

### Parent contract

`items[*].parent_branch_id` must identify a Branch that already exists and is usable at
the moment the batch request starts. A Branch created by another item in the same batch
cannot be used as a parent.

Therefore this is invalid:

```text
same batch:
root -> b1
b1   -> b1a
```

Beam Search must split work by depth. A valid multi-parent batch looks like:

```text
pre-existing b1 -> c1
pre-existing b2 -> c2
```

The target integration is one Beam depth per `emulate_actions` request.

## Admission and execution

Admission is all-or-nothing. RL validates every item in Phase A before registering any
new Branch or mutating RNG hypothesis state. If any item is invalid, the entire request
is rejected and no item is admitted.

After successful admission, Phase B registers and queues every WorkItem before one
`BranchManager.poll()` call. That call synchronously waits for every Branch dispatched
by the call to reach a terminal outcome. Worker processes may execute the queued items
in parallel.

## Response

A successfully admitted/executed batch has top-level:

```json
{
  "status": "completed",
  "branch_results": {
    "c1": { "...": "..." },
    "c2": { "...": "..." }
  }
}
```

Top-level `status = "completed"` describes completion of the batch request itself; it
does not mean every Branch succeeded.

Each requested `branch_id` appears exactly once in `branch_results`, with no missing or
extra keys. Every per-Branch result is terminal:

- `completed`: Branch execution succeeded and includes the normal decision payload.
- `partial`: allowed only if `BranchManager` actually produces a partial terminal result;
  it includes the normal decision payload.
- `faulted`: that Branch failed; sibling Branches may still be `completed`.

`queued` and `running` are not valid normal `emulate_actions` response outcomes because
`BranchManager.poll()` has already synchronously resolved the Branches it dispatched.
A missing poll result is an internal invariant violation and must surface as a fault or
exception, never as a normal `running` fallback.

For every per-Branch result, Training correlates these fields against the corresponding
request item:

```text
branch_id
parent_branch_id
rng_id
```

`completed` / `partial` responses additionally retain the existing decision-payload
validation. Any missing result, extra result, or correlation mismatch is an
`ApiProtocolError`; completion is uncertain and the exact request remains pending for
retry.

## Retry and at-most-once semantics

The existing single in-flight sequencing rules apply to the batch as a whole.
Completion-uncertain failure stores the exact serialized `emulate_actions` request and
fresh requests are blocked until that exact request is replayed.

The entire batch shares one wire `request_id`. RL's session replay handling therefore
applies at-most-once semantics to the whole batch, not independently to each item.
Training's selection audit treats each item as a separate logical selection using an
item-scoped identity equivalent to `(request_id, branch_id)`:

- first attempt: one `selection` event per item;
- exact replay: one `selection_recovery` event per item;
- replay never creates a second logical `selection` for the same item.

## Explicit non-goals for v0.7

The following are outside this contract change:

- making `BranchManager.poll()` async;
- adding a concurrency lock inside `BranchManager`;
- TCP request/response multiplexing;
- multiple Training requests in flight;
- multiple in-flight entries in the session ledger;
- removing the global handler lock;
- per-instance concurrent request handlers;
- out-of-order response handling;
- Beam frontier generation/scoring/pruning changes themselves.

Beam integration is a follow-up change that should replace frontier-by-frontier single
calls with `emulate_actions([...])`, targeting `1 Beam depth = 1 emulate_actions request`.
