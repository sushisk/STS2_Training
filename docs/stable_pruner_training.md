# Supervised Stable Frontier Pruner

This document describes the first learned search-control baseline built on the budgeted
Oracle data from PR #47.

## Scope

The model replaces only the `StableFrontierPruner` selection rule. It does **not** own:

- `PolicyModel.top_k_actions`
- continuation allocation/order
- Whole Run active-branch capacity
- dynamic beam width or stopping
- parent expansion scheduling

Those remain separate search-control mechanisms and may later move behind a broader
`SearchController`.

## Training target

The initial model is a contextual per-node linear ranker. For every stable prune frontier,
Oracle collection provides a downstream `target_value` for nodes that the wider teacher
search actually followed far enough to produce a fresh leaf/terminal outcome.

Training creates pairwise examples within the same prune frontier:

```text
x_pair = features(better_node) - features(worse_node)
y_pair = 1
```

and the symmetric negative example. Near-ties below `--min-target-gap` are ignored.
This directly matches runtime use: the learned model produces one scalar score per node,
then keeps the top-K scores.

`no_target` nodes are excluded rather than treated as negative labels. Terminal targets
receive full weight by default. `value_bootstrap` targets remain usable but are downweighted
(default `0.5`) because they are censored by the Oracle search horizon / ValueModel.

## Feature contract

`PRUNER_FEATURE_NAMES` is shared by offline training and dependency-free runtime inference.
The first schema deliberately uses only information already present in stable-prune traces:

- current ValueModel score and frontier-relative value statistics/rank
- root-action group size, group-relative value statistics, and within-group rank
- search depth / combat depth / continuation count / terminal flag
- policy rank/score availability and post-coverage rank
- structural-coverage provenance
- coarse action type
- target beam width, frontier size, remaining depth, and remaining time

Opaque `action_id` values are never used as learned identities. `root_action_id` is used
only to form within-frontier groups.

For Oracle training records, the feature named `beam_width` is populated from
`target_beam_width`, not the wider teacher search's prune K. This keeps the feature meaning
aligned with the student decision being learned.

## Train

Oracle JSONL can be collected with the runner introduced by PR #47. Train the ranker with:

```bash
python tools/train_stable_pruner.py \
  --log-dir data/combat_oracle \
  --output tools/output/stable_pruner_weights.json
```

Useful options:

```text
--val-fraction 0.1
--test-fraction 0.1
--inverse-regularization 1.0
--min-target-gap 1e-6
--terminal-weight 1.0
--bootstrap-weight 0.5
```

Splitting is by source JSONL file, not individual frontier, to reduce leakage between
nearby decisions from one collected episode. Collection workflows should therefore prefer
one episode/run per JSONL file (or otherwise preserve an episode-level file boundary).

The training-only implementation uses `numpy`/`scikit-learn`. The emitted JSON artifact
contains feature schema/version, coefficients, scaler values, training settings,
provenance, and metrics.

## Offline metrics

Each training split reports:

- pairwise accuracy / weighted pairwise accuracy
- learned Recall@K against downstream `target_value` top-K
- `ValueTopKPruner` Recall@K on the same labeled slice
- best reachable target-value regret for learned and ValueTopK selection
- mean selected target-value gap versus teacher top-K

For a completely separate held-out Oracle collection, evaluate an artifact without
retraining:

```bash
python tools/eval_stable_pruner.py \
  --weights tools/output/stable_pruner_weights.json \
  --log-dir data/combat_oracle_heldout
```

The evaluator reports the same learned-vs-`ValueTopKPruner` ranking/selection metrics and
includes the artifact hash-derived version. This makes repeated evaluations of different
artifacts unambiguous.

These are teacher-distillation metrics, not final gameplay quality. A model should not be
promoted solely because it improves pairwise accuracy.

## Runtime

Runtime inference is standard-library only:

```python
from sts2_training.decision import CombatDecisionEngine, LinearStableFrontierPruner

pruner = LinearStableFrontierPruner.from_weights_file(
    "tools/output/stable_pruner_weights.json"
)
engine = CombatDecisionEngine(client, stable_pruner=pruner)
```

The artifact SHA-256 prefix becomes the pruner `version` recorded in search traces, so
results from different learned artifacts remain distinguishable.

## Fixed-seed A/B evaluation before RL

After held-out Oracle evaluation, validate the artifact against the exact current
`ValueTopKPruner` baseline with the real emulator:

```bash
python -m sts2_training.runner.stable_pruner_ab \
  --scenario data/scenarios/slime.json \
  --weights tools/output/stable_pruner_weights.json \
  --seeds 101,102,103,104 \
  --search-mode standard \
  --output tools/output/stable_pruner_ab.json
```

Each seed is run twice from the same `CombatScenario`: once with `ValueTopKPruner` and once
with `LinearStableFrontierPruner`. Both arms receive the same scenario seed and an
independently constructed `CombatDecisionEngine` with the same Beam configuration and the
same default Policy/Value implementations. `--arm-order alternate` is the default, so the
first/second execution order alternates between seed pairs instead of systematically
favoring one arm through server warm-up or ordering effects.

The report records, per arm:

- final terminal outcome
- committed action IDs for diagnostics plus canonicalized action semantics
- number of decisions and heuristic fallbacks
- Beam reason / best value per decision
- nodes expanded and branches created
- measured Beam search milliseconds and episode elapsed time
- pruner name/version (the learned artifact version contains its hash prefix)

`action_id` is valid only inside one decision and is never compared across the two A/B
instances. For common-prefix/divergence detection the runner canonicalizes the chosen legal
action payload after removing `action_id` and `is_available`, then compares that semantic
signature. Once the signatures diverge, later states can differ, so subsequent choices are
not treated as paired counterfactuals. The pair report therefore exposes
`common_action_prefix` and `first_divergence_index`; after divergence, compare terminal
outcomes and arm-level search cost instead.

The summary reports learned/baseline wins, ties/unknown outcomes, divergence rate, mean
search cost for each arm, and learned-minus-baseline cost deltas. A gameplay promotion
decision should combine these real-emulator results with held-out Oracle regret/Recall@K;
neither one replaces the other.

Real-emulator A/B is intentionally separate from the repo-local hosted unit/contract gate.
The current CI proves deterministic code/contract behavior but does not attest an exact
Training/RL runtime pair or gameplay quality.

Only after the supervised artifact is stable under both held-out Oracle evaluation and
fixed-seed emulator A/B should the search policy be fine-tuned on trajectories induced by
its own pruning choices.

The per-node ranker is intentionally a baseline. It cannot fully model the value of a
*set* of retained nodes, especially redundancy among many branches from one root action.
Future models may therefore become root-group-aware, listwise, or explicit set selectors
before/alongside RL fine-tuning.
