# Combat Search Learning Implementation Plan

## Goal

Make combat search decisions measurable, replaceable, and learnable without changing current runtime behavior first.

The long-term search stack is intentionally split into three roles:

1. `PolicyModel`: rank/propose legal combat actions.
2. `ValueModel`: evaluate resolved stable/terminal combat states.
3. Stable pruning first, then a broader `SearchController`: decide which search branches deserve limited compute.

The first implementation step is infrastructure, not RL training.

## Current search-control boundaries

`BeamSearchEngine` currently contains several distinct forms of pruning / compute allocation:

- `PolicyModel` candidate truncation through `top_k_actions`
- Whole Run active-branch capacity planning before emulation
- continuation-frontier pruning by policy/item order while pending choices are resolved
- stable/resolved frontier pruning by `node.value` top-K

Phase 1 deliberately extracts only the last responsibility. It must not imply that all search allocation is already controlled by one pruner.

## Phase 1: Extract the stable-frontier pruning seam

Introduce a `StableFrontierPruner` interface used by `BeamSearchEngine` only when resolved/stable nodes are merged and selected for the next beam.

Provide `ValueTopKPruner` as the default implementation. It must reproduce the current stable-frontier behavior exactly, including:

- `waiting_stable` accumulation while continuation nodes remain unresolved
- merging `waiting_stable` back into the stable frontier after continuation resolution
- descending `node.value` ordering
- Python stable-sort tie ordering
- the `waiting_stable` fallback when the live beam becomes empty

This seam does **not** control `PolicyModel.top_k_actions`, continuation ordering, or Whole Run active-branch capacity allocation.

Keep structural candidate coverage in `CoverageConstrainedPolicy`; do not mix those constraints into the learned stable pruner yet.

Longer term, parent expansion, actions-per-parent, continuation allocation, dynamic beam width, stopping, and Whole Run capacity allocation belong in a broader `SearchController` abstraction.

## Phase 2: Record search traces

Add a dedicated search-training trace separate from `SelectionAudit`.

For each relevant search/pruning step, record enough information to reconstruct the decision, including:

- decision/root action identity and semantic action data
- depth and remaining search budget
- value estimate and value delta from parent
- stable/terminal/continuation state kind
- root-action group and within-group rank
- whether the node survived or was pruned
- RNG hypothesis/provenance
- search configuration and model/artifact provenance

Continuation nodes must be explicitly marked because their inherited `node.value` is not a fresh `ValueModel` estimate.

### Policy / coverage provenance

The trace must distinguish the policy's own ranking from structural coverage applied afterward. Where available, record:

- `policy_rank`: rank from the inner `PolicyModel`
- optional `policy_score` / `policy_prior` / logit
- `post_coverage_rank`: rank after `CoverageConstrainedPolicy`
- `candidate_source = policy | structural_coverage`
- whether structural coverage inserted or replaced a candidate

The collector therefore needs an observation point around the policy/coverage boundary rather than reconstructing this information from the final `ActionCandidate` list. The runtime `ActionCandidate` contract does not need to treat opaque `action_id` as a learned global identity.

## Phase 3: Add a budgeted oracle collection mode

Add a data-collection mode with a wider/deeper search budget than runtime search.

The result is a **configuration-dependent oracle estimate**, not ground truth. A wider/deeper search can still be censored by policy candidate limits, depth/time limits, active-branch capacity, and `ValueModel` bootstrap error.

Each target therefore carries metadata sufficient to interpret its quality, including:

- oracle/search budget and full search configuration
- `terminal_reached`
- `target_source = terminal | value_bootstrap`
- `censored` plus a truncation/censoring reason
- policy/value model provenance
- raw per-RNG outcomes where stochastic simulation is involved

The oracle trace should support deriving both:

- action-policy targets: estimated `Q(s, a)` for legal root actions
- pruning targets: estimated future utility of retaining each stable frontier node

### Common-RNG comparisons

Common-random-number comparisons are an **oracle-only sampling experiment**, not an assumption about ordinary beam behavior.

The current beam gives different root candidates different `rng_id` values and descendants inherit the root candidate's RNG hypothesis. Therefore merely increasing beam width/depth does not create paired-RNG comparisons.

Before enabling paired comparisons across different root actions, explicitly verify the Training/RL API contract guarantees the intended meaning of reusing one RNG hypothesis across those different actions. If verified, record the paired experiment configuration and raw per-RNG outcomes; otherwise keep ordinary independent-hypothesis collection.

## Phase 4: Bootstrap a learned stable pruner

Start with supervised imitation/distillation from oracle traces rather than policy-gradient RL from scratch.

The initial baseline uses a contextual per-node score:

```text
score = f(node features, frontier-relative features, remaining budget)
```

Then apply structural safety constraints and retain top-K nodes.

Initial targets can use reachable descendant value and oracle survivor labels. More expensive marginal-utility / leave-one-out labels should be collected selectively near the pruning boundary.

Independent per-node scoring is only the first baseline. Root-action regret is fundamentally affected by the value of the retained **set**, including redundancy between similar nodes. Later versions may therefore use root-action-group-aware, listwise, or explicit set-selection control.

## Phase 5: RL fine-tuning

After the supervised pruner is stable, fine-tune the search policy on trajectories induced by the learned controller.

The eventual objective should optimize root-decision quality under a compute budget, for example:

```text
reward = - root_action_regret - lambda * emulator_cost - mu * latency
```

This phase may later expand `StableFrontierPruner` into a `SearchController` that also controls parent expansion, actions-per-parent, continuation allocation, dynamic beam width, stopping, and Whole Run active-branch capacity allocation.

## Initial implementation PR scope

The first code PR after this plan should contain only:

- `StableFrontierPruner` abstraction
- `ValueTopKPruner` exact-behavior baseline for stable/resolved frontier pruning
- `BeamSearchEngine` integration at that single pruning seam
- search trace schema / collector hooks
- policy/coverage provenance hooks sufficient to distinguish inner ranking from structural coverage
- regression tests proving default search decisions are unchanged

The regression suite must cover at least:

- ordinary stable frontier pruning
- `waiting_stable` merge
- all-equal values / stable tie ordering
- continuation resolving back into the stable frontier
- `waiting_stable` fallback when the live beam becomes empty
- coexistence with Whole Run active-branch capacity pruning

No learned model, training dependency, or RL algorithm should be introduced in that PR.

## Acceptance criteria

1. Existing combat-search behavior remains unchanged with `ValueTopKPruner` as the default stable-frontier pruner.
2. Search-pruning decisions can be replayed/analyzed from emitted traces.
3. Trace records distinguish inner-policy ranking from post-coverage candidate selection where instrumentation is available.
4. Oracle-derived labels identify their budget/configuration, terminal-vs-bootstrap source, and censoring/truncation status rather than presenting themselves as ground truth.
5. Paired/common-RNG collection is gated on an explicitly verified API semantic contract.
6. Trace records do not treat opaque `action_id` values as global learned identities.
7. Training instrumentation remains separable from normal selection logging.
8. The Phase 1 seam is explicitly limited to stable/resolved frontier pruning; broader compute allocation remains future `SearchController` work.
