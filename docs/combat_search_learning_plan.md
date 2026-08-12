# Combat Search Learning Implementation Plan

## Goal

Make combat search pruning measurable, replaceable, and learnable without changing current runtime behavior first.

The long-term search stack is intentionally split into three roles:

1. `PolicyModel`: rank/propose legal combat actions.
2. `ValueModel`: evaluate resolved stable/terminal combat states.
3. `FrontierPruner` / later `SearchController`: decide which search branches deserve limited compute.

The first implementation step is infrastructure, not RL training.

## Phase 1: Extract the pruning seam

Introduce a `FrontierPruner` interface used by `BeamSearchEngine` after child states are scored.

Provide `ValueTopKPruner` as the default implementation. It must reproduce the current behavior exactly: rank stable frontier nodes by `node.value` and retain the best `beam_width` nodes.

Keep structural candidate coverage in `CoverageConstrainedPolicy`; do not mix those constraints into the learned pruner yet.

## Phase 2: Record search traces

Add a dedicated search-training trace separate from `SelectionAudit`.

For each pruning step, record enough information to reconstruct the decision, including:

- decision/root action identity and semantic action data
- depth and remaining search budget
- policy rank/prior where available
- value estimate and value delta from parent
- stable/terminal/continuation state kind
- root-action group and within-group rank
- whether the node survived or was pruned
- downstream best result discovered from the node when running an oracle search
- RNG hypothesis/provenance and search configuration

Continuation nodes must be explicitly marked because their inherited `node.value` is not a fresh state-value estimate.

## Phase 3: Add an oracle collection mode

Add a data-collection mode with a wider/deeper search budget than runtime search.

The oracle trace should support deriving both:

- action-policy targets: estimated `Q(s, a)` for legal root actions
- pruning targets: estimated future utility of retaining each frontier node

When alternatives are compared under stochastic simulation, use common RNG hypotheses where the API contract allows it and retain raw per-RNG outcomes in the trace.

## Phase 4: Bootstrap a learned pruner

Start with supervised imitation/distillation from oracle traces rather than policy-gradient RL from scratch.

Use a contextual per-node score:

```text
score = f(node features, frontier-relative features, remaining budget)
```

Then apply structural safety constraints and retain top-K nodes.

Initial targets can use reachable descendant value and oracle survivor labels. More expensive marginal-utility / leave-one-out labels should be collected selectively near the pruning boundary.

## Phase 5: RL fine-tuning

After the supervised pruner is stable, fine-tune the search policy on trajectories induced by the learned controller.

The eventual objective should optimize root-decision quality under a compute budget, for example:

```text
reward = - root_action_regret - lambda * emulator_cost - mu * latency
```

This phase may later expand `FrontierPruner` into a `SearchController` that also controls parent expansion, actions-per-parent, dynamic beam width, and stopping.

## Initial implementation PR scope

The first code PR after this plan should contain only:

- `FrontierPruner` abstraction
- `ValueTopKPruner` exact-behavior baseline
- `BeamSearchEngine` integration
- search trace schema / collector hooks
- regression tests proving default search decisions are unchanged
- trace tests covering stable, terminal, and continuation nodes

No learned model, training dependency, or RL algorithm should be introduced in that PR.

## Acceptance criteria

1. Existing combat-search behavior remains unchanged with the default pruner.
2. Search-pruning decisions can be replayed/analyzed from emitted traces.
3. Trace records do not treat opaque `action_id` values as global learned identities.
4. Training instrumentation remains separable from normal selection logging.
5. The seam is sufficient to add a heuristic or learned pruner without modifying core beam-search traversal logic.
