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

For Combat, `simulation_options.max_time_ms` is a **per-Branch execution timeout**, not
a wall-clock deadline for the entire batch. `BranchManager.poll()` keeps at most one
outstanding request on each worker. Branches beyond the worker count remain
coordinator-side `queued` until a compatible worker becomes free, and their timeout does
not start until they are actually dispatched. Completion on another worker never extends
an executing Branch's deadline. If a worker times out, only that worker's executing
Branch receives `task_timeout`; the worker is respawned and a queued tail Branch is then
dispatched with its own fresh deadline. Consequently total batch wall-clock time may
exceed `max_time_ms` when a batch is larger than the worker count.

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
    "stop_condition": "next_decision",
    "max_time_ms": 60000
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

### Batch-size capability

A single `emulate_actions` request cannot contain more Branches than RL's configured
`BranchManager.max_branches` capacity. Combat publishes that instance-specific limit in
the completed `start_instance` response as the positive integer
`max_emulate_actions_items`:

```json
{
  "status": "completed",
  "instance_id": "inst-001",
  "max_emulate_actions_items": 64
}
```

The standard Combat configuration uses 64, but callers must not assume that value: the
server may be configured with a smaller or larger capacity. Training caches the published
limit for the active instance and rejects an `emulate_actions` request that exceeds it,
so Beam can chunk a wide frontier deterministically before sending it. This capability is
the configured maximum batch size, not a claim about the momentary number of free Branch
slots; RL still performs the authoritative active-capacity admission check.

The Beam integration target is therefore "one depth = one or more bounded batch
requests", not an unconditional one-request-per-depth guarantee.

## Admission and execution

Admission is all-or-nothing. RL validates every item in Phase A before registering any
new Branch or mutating RNG hypothesis state. If any item is invalid, the entire request
is rejected and no item is admitted.

After admission, Phase B prepares all WorkItems before committing internal Branch
records. Heterogeneous-parent Branches are then registered as one manager batch before a
single `BranchManager.poll()` call. If coordinator-side preparation, submission,
dispatch, result collection, or response finalization raises, all internal Branches from
that batch are cancelled/released, batch bookkeeping is removed, and RNG allocation
state is restored. Any public branch ID already registered before such an unexpected
failure remains burned (non-reusable) but is quarantined and cannot become a parent or
execute later.

`BranchManager.poll()` synchronously waits for every Branch admitted by the call to reach
a terminal outcome. Worker processes may execute up to one Branch per worker in
parallel; additional Branches stay coordinator-side `queued` until a worker is free.

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
`BranchManager.poll()` has already synchronously resolved the Branches admitted for that
batch. A missing poll result is a coordinator invariant violation. RL must fail the whole
Phase B transaction and run the same cancel/release/bookkeeping/RNG quarantine path used
for other unexpected coordinator failures; it must not manufacture a normal per-Branch
`faulted` result under a top-level `completed` response.

For every normal per-Branch result, Training correlates these fields against the
corresponding request item:

```text
branch_id
parent_branch_id
rng_id
```

`completed` / `partial` responses additionally retain the existing decision-payload
validation. Any missing result, extra result, or correlation mismatch observed by
Training is an `ApiProtocolError`; completion is uncertain and the exact request remains
pending for retry.

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

Because the protocol is single-in-flight, Training only retains replay identities for
the immediately current/previous request. When a different `request_id` is observed,
older selection identities are discarded; audit replay bookkeeping is therefore bounded
by one batch rather than total speculative selections over the search.

## Version rollout and deployment compatibility

DTO v0.7 is a deliberate **hard cutover**, not a rolling-compatible extension of v0.6.
An RL endpoint and a Training client participating in v0.7 must therefore be deployed or
activated as one lockstep compatibility unit. A deployment must not route v0.7 Training
to a v0.6 RL endpoint, or v0.6 Training to a v0.7-only RL endpoint.

If an environment cannot guarantee that lockstep activation, it must add an explicit
version-negotiation or dual-version compatibility mechanism before adopting v0.7. This
PR pair does not claim mixed-version interoperability.

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
calls with bounded `emulate_actions([...])` chunks, targeting one Beam depth per one or
more batch requests.
