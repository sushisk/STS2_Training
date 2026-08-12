# Combat Search Learning Implementation Plan

## Goal

Make combat search decisions measurable, replaceable, and learnable without changing current runtime behavior first.

The long-term search stack is intentionally split into three roles:

1. `PolicyModel`: rank/propose legal combat actions.
2. `ValueModel`: evaluate resolved stable/terminal combat states.
3. Stable pruning first, then a broader `SearchController`: decide which search branches deserve limited compute.

Phase 1 now establishes the instrumentation and replacement seam. Training the learned components remains a later phase.

## Current search-control boundaries

`BeamSearchEngine` contains several distinct forms of pruning / compute allocation:

- `PolicyModel` candidate truncation through `top_k_actions`
- Whole Run active-branch capacity planning before emulation
- continuation-frontier pruning by policy/item order while pending choices are resolved
- stable/resolved frontier pruning by `node.value` top-K

Phase 1 extracts only the last responsibility. It does not imply that all search allocation is already controlled by one pruner.

## Phase 1: Stable-frontier pruning seam and trace infrastructure

`StableFrontierPruner` is used by `BeamSearchEngine` only when an already ordered resolved/stable frontier is selected for the next beam.

`BeamSearchEngine`, not the pruner, owns `waiting_stable` accumulation, merge order, and fallback lifecycle. This keeps the pruner a pure selection seam and makes parity tests and later learned implementations simpler.

`ValueTopKPruner` is the default implementation and reproduces the current stable-frontier behavior:

- descending `node.value` ordering
- Python stable-sort tie ordering
- top-K selection after `waiting_stable` has been merged by the engine
- the same selection rule for the `waiting_stable` fallback

This seam does **not** control `PolicyModel.top_k_actions`, continuation ordering, or Whole Run active-branch capacity allocation.

Structural candidate coverage remains in `CoverageConstrainedPolicy`; it is not mixed into the learned stable pruner.

Longer term, parent expansion, actions-per-parent, continuation allocation, dynamic beam width, stopping, and Whole Run capacity allocation belong in a broader `SearchController` abstraction.

## Phase 2: Search traces

Search-training traces remain separate from `SelectionAudit`.

Phase 1 trace records include replay/grouping metadata from the start:

- `search_id`
- proposal/prune step IDs
- `node_id` and `parent_node_id`
- `frontier_index_before_prune`
- requested `K`
- pruner name/version
- depth and remaining search budget
- value estimate
- stable/terminal state information
- root-action identity
- RNG hypothesis/provenance
- semantic action payloads

The stable-prune trace records **every node presented to the runtime pruner**, not only survivors. This creates an explicit observation point from which a later oracle collector can evaluate branches the runtime pruner dropped.

### Policy / coverage provenance

The trace distinguishes the policy's own ranking from structural coverage applied afterward. It records:

- `policy_rank`
- optional `policy_score` / prior / logit when the policy exposes one
- `post_coverage_rank`
- `candidate_source = policy | structural_coverage`
- the full legal-action set visible at that decision

Keeping the full legal-action set is important because a proposal trace must distinguish actions that were illegal from actions that were legal but never evaluated because of policy/top-K filtering. Opaque `action_id` values remain decision-local API identifiers rather than learned global identities.

## Phase 3: Budgeted oracle collection

Add a data-collection mode with a wider/deeper search budget than runtime search.

The result is a **configuration-dependent oracle estimate**, not ground truth. A wider/deeper search can still be censored by policy candidate limits, depth/time limits, active-branch capacity, and `ValueModel` bootstrap error.

Each target therefore carries metadata sufficient to interpret its quality, including:

- oracle/search budget and full search configuration
- `terminal_reached`
- `target_source = terminal | value_bootstrap`
- `censored` plus a truncation/censoring reason
- policy/value model provenance
- raw per-RNG outcomes where stochastic simulation is involved

### Root-action targets and selection bias

When the oracle still uses a policy candidate limit, its action-policy labels are estimates for **evaluated legal root actions**, not automatically for every legal root action.

The collector should support two explicit modes:

1. candidate-limited mode: evaluate only actions admitted by the configured proposal policy and mark other legal root actions as policy-censored/no-target;
2. exhaustive-root mode: when budget permits, emulate every available root action before deeper search so root-action comparison is not initially filtered by the runtime policy.

The proposal trace already retains the full legal-action set needed to distinguish these cases.

### Runtime-pruned stable nodes as oracle observation points

Pruning targets must not simply imitate the runtime pruner's survivor set. For each stable frontier, the oracle collector should be able to continue evaluating nodes that the runtime pruner would have dropped.

The Phase 1 stable-prune trace therefore captures the complete pre-prune frontier and survivor flag. A later oracle pass may attach a downstream target to any of those nodes. If a node cannot be evaluated within the oracle budget, it must remain `censored/no_target`; being runtime-pruned is not itself a negative label.

This is the key protection against turning the current weak pruning rule into its own teacher.

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

Initial targets can use reachable descendant value and oracle survivor/utility labels. More expensive marginal-utility / leave-one-out labels should be collected selectively near the pruning boundary.

Independent per-node scoring is only the first baseline. Root-action regret is fundamentally affected by the value of the retained **set**, including redundancy between similar nodes. Later versions may therefore use root-action-group-aware, listwise, or explicit set-selection control.

## Phase 5: RL fine-tuning

After the supervised pruner is stable, fine-tune the search policy on trajectories induced by the learned controller.

The eventual objective should optimize root-decision quality under a compute budget, for example:

```text
reward = - root_action_regret - lambda * emulator_cost - mu * latency
```

This phase may later expand `StableFrontierPruner` into a `SearchController` that also controls parent expansion, actions-per-parent, continuation allocation, dynamic beam width, stopping, and Whole Run active-branch capacity allocation.

## Phase 1 implementation status

The current PR implements:

- `StableFrontierPruner`
- `ValueTopKPruner` as the default exact stable-frontier baseline
- `BeamSearchEngine` integration at only the stable/resolved pruning seam
- search trace schema and collector hooks
- replay/grouping/order metadata
- policy/coverage provenance
- full legal-action proposal snapshots
- complete pre-prune stable-frontier snapshots with survivor flags
- focused regression tests for stable ordering and instrumentation behavior

No learned model, training dependency, or RL algorithm is introduced in Phase 1.

## Acceptance criteria

1. Existing combat-search behavior remains unchanged with `ValueTopKPruner` as the default stable-frontier pruner.
2. Search-pruning decisions can be replayed/analyzed from emitted traces.
3. Trace records distinguish inner-policy ranking from post-coverage candidate selection.
4. Legal-but-policy-censored root actions can be distinguished from evaluated actions, and a later exhaustive-root oracle mode is supported by the trace contract.
5. Runtime-pruned stable nodes remain oracle observation candidates; an unevaluated node is `censored/no_target`, not a negative target.
6. Oracle-derived labels identify their budget/configuration, terminal-vs-bootstrap source, and censoring/truncation status rather than presenting themselves as ground truth.
7. Paired/common-RNG collection is gated on an explicitly verified API semantic contract.
8. Trace records do not treat opaque `action_id` values as global learned identities.
9. Training instrumentation remains separable from normal selection logging.
10. The Phase 1 seam is explicitly limited to stable/resolved frontier pruning; broader compute allocation remains future `SearchController` work.
