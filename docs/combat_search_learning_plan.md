# Combat Search Learning Implementation Plan

## Goal

Make Combat search measurable and progressively learnable while keeping runtime behavior explicit.

The long-term stack has three roles:

1. `PolicyModel`: rank/propose legal Combat actions.
2. `ValueModel`: evaluate resolved stable/terminal Combat states.
3. Search control: first `StableFrontierPruner`, later a broader `SearchController` that allocates compute.

This PR implements both the stable-pruning/trace foundation and the first budgeted Oracle data-collection path. Learned models and RL remain later phases.

The dependency contract is intentionally one-way: #47 owns the stable-pruning public data/API contract, later supervised learned-pruner work consumes it, and later RL work consumes the learned artifact/feature contract. Downstream work must not depend on private `BeamNode` fields or require this seam to grow retroactively.

## Score terminology

Four distinct "score" concepts recur across `beam_search.py`, `candidate_coverage.py`, `stable_pruner.py`, and `oracle_search.py`/`oracle_value_logging.py`. Each is exposed as a `@property` documented `"""Canonical name for ..."""` over an underlying raw field, so code can be read without memorizing which raw field belongs to which role. Do not introduce a new raw field name for a concept one of these four already covers.

- **`state_score`** — the `ValueModel`'s evaluation of one already-resolved (stable/terminal) Combat state. Canonical name for `value` on `BeamNode`/`StablePruneNodeView`. Never valid for a pending/continuation node, whose value is inherited/stale (see `StablePruneNodeView.value`'s own contract in "Public stable-pruning contract v1" below).
- **`action_score`** — a candidate action's score *before* it is simulated/resolved, i.e. the policy's own prior/ranking score for that action. Canonical name for `policy_score` (`BeamNode`, `RankedCandidate` in `candidate_coverage.py`, `StablePruneNodeView`). Pairs with `action_rank`, the canonical name for the matching pre-coverage `policy_rank`. `None` when the policy exposes no score for that candidate.
- **`node_score`** — the score attributed to a search-tree node as a whole. In `BeamSearchResult`/`SearchTraceEvent` this is just the winning node's `state_score` (canonical name for `best_value`). In Oracle aggregation (`oracle_search.py`'s `OracleRngOutcome`/`RootActionOracleTarget`) it is broader: an RNG hypothesis's attributed outcome, or a root action's aggregated `estimated_q` across RNG hypotheses — not necessarily one single state's direct `ValueModel` output.
- **`target_node_score`** — the descendant-derived node-score *used as a supervised training label* (`StableNodeOracleTarget`/`RootActionValueSample` in `oracle_search.py`/`oracle_value_logging.py`, canonical name for `target_value`). Deliberately distinct from `node_score`: a stable-pruning target is bootstrapped from a later/deeper fresh leaf outcome (see "Outcome semantics" below), not from the node's own immediately-computed score, so `target_node_score` and that same node's `node_score` are not interchangeable and must not be conflated when building training data.

`action_rank`/`policy_rank` and `behavior_frontier_scores` (`pruner_rl.py`, the RL fine-tuning behavior policy's stochastic top-K sampling scores) are related but are not part of this four-term set — they rank or drive sampling rather than evaluate a state or node.

## Search-control boundaries

`BeamSearchEngine` currently has several separate allocation/pruning mechanisms:

- `PolicyModel` candidate truncation through `top_k_actions`
- Whole Run active-branch capacity planning before emulation
- continuation-frontier pruning by policy/item order
- stable/resolved frontier pruning

Only the final responsibility is abstracted by `StableFrontierPruner`. Policy top-K, continuation allocation, and Whole Run capacity remain outside that seam until a future `SearchController` intentionally subsumes them.

## Phase 1: stable pruning and trace infrastructure — implemented

`StableFrontierPruner` receives an already ordered stable/resolved frontier. `BeamSearchEngine` retains ownership of `waiting_stable`, continuation handling, merge order, capacity planning, internal node identity, and the mapping from public survivor indices back to internal `BeamNode` objects.

### Public stable-pruning contract v1

`STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION = 1` freezes the public pruning input as the immutable `StablePruneNodeView` with exactly these fields:

- `value`
- `root_action_id`
- `depth`
- `combat_depth`
- `continuation_steps`
- `terminal`
- `action_type`
- `policy_rank`
- `policy_score`
- `post_coverage_rank`
- `candidate_source`

Field semantics are part of the schema contract, not merely implementation details:

- `value` is the finite current score of a stable/resolved/terminal node. Continuation nodes whose value is inherited/stale are forbidden at this seam.
- `root_action_id` is an opaque grouping key valid only inside one search. It may group nodes derived from the same root action but must never become a global learned identity.
- `depth` is current Beam transition depth.
- `combat_depth` counts non-continuation Combat actions.
- `continuation_steps` is the current continuation-safety counter maintained by Beam Search.
- `terminal` says whether this stable/resolved node is terminal.
- `action_type` is only the coarse semantic type of the action that produced the node; the full action payload is private.
- `policy_rank` is the inner-policy **0-based** rank. It is `None` when structural coverage inserted an action that had no inner-policy rank.
- `policy_score` contains a finite float when the policy exposes one, otherwise `None`.
- `post_coverage_rank` is the **0-based** rank after structural coverage; `None` is reserved for synthetic/legacy nodes whose provenance was unavailable.
- `candidate_source` is currently `policy` or `structural_coverage`; `None` is reserved for synthetic/legacy nodes whose provenance was unavailable.

A field add/remove/rename or a change to any of these meanings requires incrementing `STABLE_PRUNE_NODE_VIEW_SCHEMA_VERSION`.

The seam deliberately excludes `BeamNode`, masked DTOs, full action payloads, branch/node/decision-point IDs, `rng_id`, `branch_log`, and Whole Run capacity state. Those remain engine/trace internals.

The selector API is:

```python
class StableFrontierPruner:
    def select(
        self,
        frontier: Sequence[StablePruneNodeView],
        *,
        k: int,
        context: StablePruneContext,
    ) -> list[int]: ...
```

The input sequence order is the authoritative frontier order established by `BeamSearchEngine` after stable/waiting merge. Returned values are unique integer indices into that sequence, in `[0, len(frontier))`, with at most `k` entries. **The returned index order itself is survivor order.** `BeamSearchEngine` maps those indices back to private `BeamNode` objects in exactly that order. Duplicate, negative, out-of-range, bool/non-int, non-list, or `>k` selections fail fast.

`ValueTopKPruner` is the default baseline. It performs a descending value sort over frontier indices and relies on Python's stable sorting, so equal values retain authoritative frontier order. This preserves historical ordinary, tie, `waiting_stable` merge/fallback, continuation-to-stable, and Whole Run interaction behavior while changing only the public selection identity from node objects to indices.

`StablePruneContext` is also a fixed public contract. `search_id` and `prune_step_id` identify the invocation; `phase` labels the stable-pruning phase; `beam_width` is the invocation target K/runtime beam width; `max_depth` and `depths_completed` retain current Beam budget semantics; and `remaining_time_ms` is a finite non-negative float or `None`. Normal runtime calls use `k == context.beam_width`.

### Runtime/replay parity

Trace objects expose the canonical reconstruction helpers:

```python
StablePruneNodeTrace.to_prune_view() -> StablePruneNodeView
StablePruneTrace.node_views() -> tuple[StablePruneNodeView, ...]
StablePruneTrace.to_prune_context(*, beam_width: int | None = None) -> StablePruneContext
```

For each runtime prune invocation, `node_views()[i]` is field-for-field equal to the runtime `frontier[i]` view, `nodes[i].frontier_index_before_prune == i`, and `nodes[i].kept` records membership in the selected index set. `to_prune_context()` defaults `beam_width` to trace `k`; offline counterfactual replay may override only `beam_width` (for example with `target_beam_width`) while all other context fields remain identical.

Search-training traces are separate from `SelectionAudit` and include:

- `search_id`, proposal/prune step IDs
- trace-only `node_id` / `parent_node_id`
- pre-prune frontier order and K
- pruner name/version
- depth/budget metadata
- value/state/action/root-action/RNG metadata
- the complete stable frontier before pruning and survivor flags
- full legal-action snapshots
- inner-policy rank/optional score versus post-coverage rank/source
- one `search_end` completion event for every normally completed traced Beam search

Pending continuation values are explicitly inherited/stale and are not valid Value targets or public stable-pruner inputs.

The existing Oracle JSONL record schema remains **v3** for this abstraction: `StablePruneNodeTrace` already persists every field required to reconstruct node-view schema v1. A later persisted-information change should bump the Oracle record schema when warranted, but adding this public view alone does not.

## Phase 2: budgeted Oracle collection — implemented

`BudgetedOracleCollector` runs a wider/deeper Beam search on the current root Decision and produces a **configuration-dependent estimate**, never a claim of ground truth.

Default training budget is intentionally larger than runtime (`beam_width=32`, `top_k_actions=8`, `max_depth=4`), while `target_beam_width` represents the cheaper runtime-style K whose missed branches we want to study. All values remain configurable.

When constructed with `BudgetedOracleCollector.from_beam_engine()`, both the teacher policy and teacher `ValueModel` are deep-copied from the runtime engine. Collection therefore cannot advance runtime policy/value RNG, counters, caches, stochastic inference, or other mutable model state. The Beam branch/RNG allocator namespace is intentionally shared between Oracle and runtime searches so wire-level RNG hypothesis IDs are never reused on the same instance. If either model cannot be copied safely, construction fails closed and callers must provide independent teacher instances explicitly.

### Root actions

The collector supports two explicit modes:

1. **Exhaustive root** (default): request every available root action before deeper search. The collector fails closed if the policy cannot return all requested legal actions.
2. **Policy-limited root**: retain policy candidate filtering and mark legal-but-unexamined root actions `censored/no_target` with `policy_candidate_limit`.

This prevents silently calling a policy-filtered subset `Q(s,a) for all legal actions`.

### Stable-pruning targets

A wide Oracle frontier can contain nodes that a narrower runtime `ValueTopKPruner` would have removed. The target record therefore stores both:

- whether the wide Oracle actually retained the node, and
- whether the narrower `target_beam_width` ValueTopK baseline would have retained it.

If such a counterfactual node later reaches a useful leaf/terminal result, that downstream result becomes a pruning target. If the Oracle itself cuts a node before follow-up, it receives `no_target` / `oracle_pruned_before_followup`; its current Value is **not** reused as a label. This avoids simply teaching the current pruner to imitate itself.

### Outcome semantics

Targets use only fresh **leaf outcomes** from the Oracle trace. An intermediate stable state that was later expanded is not treated as final Q merely because its current `ValueModel` estimate is high.

Target source is explicit:

- `terminal`: the selected leaf reached a terminal state **and** the Value implementation explicitly supplied an exact terminal utility through `ValueModel.exact_terminal_utility()`
- `value_bootstrap`: the fresh leaf uses `ValueModel`; this also covers a terminal state when the Value implementation does not opt in to the exact-terminal-utility contract
- `mixed`: root-action aggregation contains both kinds
- `no_target`: no valid fresh leaf was observed

A terminal DTO is therefore not enough by itself to promote an arbitrary learned Value prediction into an uncensored terminal target. Each target separately carries `terminal_reached`, `censored`, and a censor/truncation reason. Root targets retain per-RNG outcomes before aggregation.

### RNG policy

Ordinary collection uses independent RNG hypotheses, matching current Beam semantics. Cross-action common-RNG sampling is deliberately rejected by `OracleCollectionConfig` until the Training/RL API explicitly guarantees the intended meaning of reusing the same RNG hypothesis across different actions.

## Phase 3: executable data collection — implemented

`OracleEpisodeRunner` collects one Oracle record **before each runtime commit**. The committed action is still chosen by the normal `CombatDecisionEngine`, not by the expensive teacher search. Together with teacher policy/Value state isolation, this keeps the visited-state distribution aligned with the runtime policy/search stack we intend to improve.

Records are written by `OracleJsonlWriter` to a dedicated JSONL file, one root Decision per line. Each record contains:

- raw masked `masked_emulator_dto` for future re-featurization
- Oracle search result and budget metadata
- root-action targets
- stable-pruning targets
- full search trace
- teacher outer policy, inner policy, structural-coverage wrapper, Value, pruner, RNG, and optional Training commit provenance
- JSON-safe teacher policy/value configuration metadata supplied through `oracle_provenance()` (for example heuristic weights, checkpoint/version/config hashes, or RNG-state identity)

Provenance is captured by `OracleCollectionResult` at the start of each collection pass, not inferred from the runtime commit engine, so a deliberately different or stateful teacher configuration is recorded correctly. Implementations that return non-JSON-safe provenance fail closed rather than silently producing ambiguous dataset lineage.

This remains separate from `SelectionAudit`.

Example:

```bash
python -m sts2_training.runner.oracle_collection \
  --scenario combat.json \
  --output data/combat_oracle.jsonl \
  --oracle-beam-width 32 \
  --oracle-top-k 8 \
  --oracle-depth 4 \
  --target-beam-width 8
```

`--policy-limited-root` disables exhaustive-root collection. `--oracle-time-budget-ms` can impose a teacher-search latency cap. `--max-decisions` is a deliberate data-collection cap rather than an episode-quality metric.

## Phase 4: supervised learned stable pruner — downstream

The first learned pruner should be bootstrapped from the Oracle records rather than from policy-gradient RL. It consumes only `StablePruneNodeView + StablePruneContext`; it must not reconstruct features from `BeamNode`, DTOs, action payloads, or trace identity fields.

Initial baseline:

```text
score = f(node features, frontier-relative features, remaining budget)
```

Then retain top-K subject to structural safety constraints. Independent per-node scoring is only a baseline; later training may support root-action-group-aware/listwise/set-selection objectives because the value of a retained set depends on redundancy and diversity.

Training data should exclude `no_target` examples from value/ranking labels while retaining them for censoring diagnostics. Exact terminal targets should generally receive greater trust than Value-bootstrap targets.

## Phase 5: RL fine-tuning — later downstream

After supervised search control is stable, optimize trajectories induced by the learned controller. Stochastic survivor selection still uses the same `StablePruneNodeView` input and index-based policy action; RL-specific trajectory/update state must not widen this #47 seam.

The eventual objective should capture decision quality under compute cost, for example:

```text
reward = - root_action_regret - lambda * emulator_cost - mu * latency
```

If control later needs parent expansion, actions-per-parent, continuation allocation, dynamic beam width, stopping, or Whole Run capacity allocation, that is a separate `SearchController` design rather than a request to add private `BeamNode` fields to `StablePruneNodeView`.

## Current PR acceptance criteria

1. `StableFrontierPruner` receives only immutable `StablePruneNodeView` objects and returns ordered `list[int]` survivor indices.
2. `BeamSearchEngine` validates returned indices and maps them to private `BeamNode` objects without changing survivor order.
3. Invalid duplicate, negative, out-of-range, bool/non-int, non-list, or `>k` selections fail fast.
4. `ValueTopKPruner` leaves default Combat-search decisions and stable tie ordering unchanged, including waiting/continuation/Whole Run interactions.
5. Runtime pruning views and `StablePruneTrace.node_views()` are field-for-field identical; context is reconstructible with a beam-width override.
6. Continuation/inherited-value nodes never enter the stable-pruning public seam.
7. `StablePruneNodeView`, node-view schema version, context, selector, and trace helpers are public exports owned by #47.
8. Policy ranking and structural coverage provenance are distinguishable.
9. Legal-but-policy-censored root actions are distinct from evaluated actions; exhaustive-root mode exists.
10. A wide Oracle can produce downstream targets for nodes a cheaper runtime K would discard.
11. Oracle-pruned/unobserved branches are `censored/no_target`, never implicit negative labels.
12. Intermediate expanded Value estimates are not treated as final Oracle Q outcomes.
13. Terminal source requires an explicit exact-terminal-utility contract; arbitrary terminal Value predictions remain censored bootstrap targets.
14. Common-RNG cross-action sampling remains disabled pending an explicit API semantic guarantee.
15. Oracle JSONL remains schema v3 for node-view v1 and retains raw masked DTOs for re-featurization.
16. Oracle collection commits actions using the runtime engine and does not advance runtime policy or ValueModel state, preserving the runtime-induced state distribution.
17. JSONL provenance identifies the actual teacher inner/wrapper policy and Value class plus JSON-safe configuration metadata rather than inferring it from the runtime engine.
18. No learned model or RL dependency is introduced by this PR.
