# Oracle branch/RNG allocator ownership

`BudgetedOracleCollector` and `BeamSearchEngine` must not reuse branch/RNG identities on the same live Training/RL instance unless the wire contract explicitly permits that reuse.

## Supported runtime-coupled construction

When Oracle collection runs beside a runtime `BeamSearchEngine` on the same client/instance, prefer:

```python
oracle = BudgetedOracleCollector.from_beam_engine(runtime_engine)
```

That constructor deliberately shares the runtime engine's `BranchIdAllocator`. The allocator therefore remains monotonic across sequential Oracle -> runtime -> Oracle searches, while the Oracle policy and ValueModel are still independent copies so teacher inference cannot mutate runtime model state.

## Direct construction

Direct `BudgetedOracleCollector(...)` construction owns an independent allocator namespace by default. This is safe when the collector also owns a disjoint instance/namespace. If a directly constructed collector is used against the same live instance as another search engine, the caller is responsible for supplying a shared or otherwise non-overlapping `branch_allocator`.

Do not assume that a released branch or RNG hypothesis ID may be reused safely on the same server instance. The current contract intentionally fails on the conservative side until Training/RL documents stronger reuse semantics.

The hosted regression suite includes an actual Oracle -> runtime -> Oracle search sequence and verifies that the wire-level `rng_id` values are distinct across all three searches.
