# Combat Search Learning Implementation Plan

## Goal

Make Combat search measurable and progressively learnable while keeping runtime behavior explicit.

The long-term stack has three roles:

1. `PolicyModel`: rank/propose legal Combat actions.
2. `ValueModel`: evaluate resolved stable/terminal Combat states.
3. Search control: first `StableFrontierPruner`, later a broader `SearchController` that allocates compute.

This PR now implements both the stable-pruning/trace foundation and the first budgeted Oracle data-collection path. Learned models and RL remain later phases.

## Search-control boundaries

`BeamSearchEngine` currently has several separate allocation/pruning mechanisms:

- `PolicyModel` candidate truncation through `top_k_actions`
- Whole Run active-branch capacity planning before emulation
- continuation-frontier pruning by policy/item order
- stable/resolved frontier pruning

Only the final responsibility is abstracted by `StableFrontierPruner`. Policy top-K, continuation allocation, and Whole Run capacity remain outside that seam until a future `SearchController` intentionally subsumes them.

## Phase 1: stable pruning and trace infrastructure — implemented

`StableFrontierPruner` receives an already ordered stable/resolved frontier and selects at most K nodes. `BeamSearchEngine` retains ownership of `waiting_stable`, continuation handling, merge order, and capacity planning.

`ValueTopKPruner` is the default baseline. It preserves the historical behavior: descending `node.value`, Python stable-sort tie ordering, and the same rule after `waiting_stable` merge/fallback.

Search-training traces are separate from `SelectionAudit` and include:

- `search_id`, proposal/prune step IDs
- `node_id` / `parent_node_id`
- pre-prune frontier order and K
- pruner name/version
- depth/budget metadata
- value/state/action/root-action/RNG metadata
- the complete stable frontier before pruning and survivor flags
- full legal-action snapshots
- inner-policy rank/optional score versus post-coverage rank/source

Pending continuation values are explicitly marked inherited/stale and are not valid Value targets.

## Phase 2: budgeted Oracle collection — implemented

`BudgetedOracleCollector` runs a wider/deeper Beam search on the current root Decision and produces a **configuration-dependent estimate**, never a claim of ground truth.

Default training budget is intentionally larger than runtime (`beam_width=32`, `top_k_actions=8`, `max_depth=4`), while `target_beam_width` represents the cheaper runtime-style K whose missed branches we want to study. All values remain configurable.

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

- `terminal`: the selected leaf reached a terminal state
- `value_bootstrap`: search stopped at a nonterminal leaf and uses `ValueModel`
- `mixed`: root-action aggregation contains both kinds
- `no_target`: no valid fresh leaf was observed

Each target carries `terminal_reached`, `censored`, and a censor/truncation reason. Root targets retain per-RNG outcomes before aggregation.

### RNG policy

Ordinary collection uses independent RNG hypotheses, matching current Beam semantics. Cross-action common-RNG sampling is deliberately rejected by `OracleCollectionConfig` until the Training/RL API explicitly guarantees the intended meaning of reusing the same RNG hypothesis across different actions.

## Phase 3: executable data collection — implemented

`OracleEpisodeRunner` collects one Oracle record **before each runtime commit**. The committed action is still chosen by the normal `CombatDecisionEngine`, not by the expensive teacher search. This keeps the visited-state distribution aligned with the runtime policy/search stack we intend to improve.

Records are written by `OracleJsonlWriter` to a dedicated JSONL file, one root Decision per line. Each record contains:

- raw masked `masked_emulator_dto` for future re-featurization
- Oracle search result and budget metadata
- root-action targets
- stable-pruning targets
- full search trace
- policy/value/pruner provenance and optional Training commit

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

## Phase 4: supervised learned stable pruner — next

The first learned pruner should be bootstrapped from the Oracle records rather than from policy-gradient RL.

Initial baseline:

```text
score = f(node features, frontier-relative features, remaining budget)
```

Then retain top-K subject to structural safety constraints. Independent per-node scoring is only a baseline; later training should support root-action-group-aware/listwise/set-selection objectives because the value of a retained set depends on redundancy and diversity.

Training data should exclude `no_target` examples from value/ranking labels while retaining them for censoring diagnostics. Terminal targets should generally receive greater trust than Value-bootstrap targets.

## Phase 5: RL fine-tuning — later

After supervised search control is stable, optimize trajectories induced by the learned controller. The eventual objective should capture decision quality under compute cost, for example:

```text
reward = - root_action_regret - lambda * emulator_cost - mu * latency
```

The abstraction can then grow from stable pruning into `SearchController`: parent expansion, actions-per-parent, continuation allocation, dynamic beam width, stopping, and Whole Run capacity allocation.

## Current PR acceptance criteria

1. `ValueTopKPruner` leaves default Combat-search decisions unchanged.
2. Stable-pruning decisions are replayable from traces with deterministic grouping/order metadata.
3. Policy ranking and structural coverage provenance are distinguishable.
4. Legal-but-policy-censored root actions are distinct from evaluated actions; exhaustive-root mode exists.
5. A wide Oracle can produce downstream targets for nodes a cheaper runtime K would discard.
6. Oracle-pruned/unobserved branches are `censored/no_target`, never implicit negative labels.
7. Intermediate expanded Value estimates are not treated as final Oracle Q outcomes.
8. Oracle labels identify terminal/bootstrap source, budget, and censoring rather than presenting themselves as ground truth.
9. Common-RNG cross-action sampling remains disabled pending an explicit API semantic guarantee.
10. Oracle JSONL is separate from normal selection logging and retains raw masked DTOs for re-featurization.
11. Oracle collection commits actions using the runtime engine, preserving the runtime-induced state distribution.
12. No learned model or RL dependency is introduced by this PR.
