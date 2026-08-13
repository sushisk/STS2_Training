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
Oracle collection provides a downstream `target_value` (canonical name: `target_node_score`
— see `combat_search_learning_plan.md`'s "Score terminology" section) for nodes that the
wider teacher search actually followed far enough to produce a fresh leaf/terminal outcome.

Training creates pairwise examples within the same prune frontier:

```text
x_pair = features(better_node) - features(worse_node)
y_pair = 1
```

and the symmetric negative example. Near-ties below `--min-target-gap` are ignored.
This directly matches runtime use: the learned model produces one scalar score per node,
then keeps the top-K scores.

`no_target` nodes are excluded from pairwise labels rather than treated as negative labels,
but remain present in the runtime-equivalent frontier used by selection evaluation.
Terminal targets receive full weight by default. `value_bootstrap` targets remain usable
but are downweighted (default `0.5`) because they are censored by the Oracle search horizon /
ValueModel.

## Feature contract

`PRUNER_FEATURE_NAMES` is shared by offline training and dependency-free runtime inference.
Feature schema v2 uses only information with matching student/runtime semantics:

- current ValueModel score and frontier-relative value statistics/rank
- root-action group size, group-relative value statistics, and within-group rank
- search depth / combat depth / continuation count / terminal flag
- policy rank/score availability and post-coverage rank
- structural-coverage provenance
- coarse action type
- target/runtime beam width and frontier size

Opaque `action_id` values are never used as learned identities. `root_action_id` is used
only to form within-frontier groups.

For Oracle training records, the feature named `beam_width` is populated from
`target_beam_width`, not the wider teacher search's prune K. Feature schema v2 deliberately
excludes remaining-depth and remaining-time features: Oracle v3 records the wider teacher
budget, so those values cannot be reconstructed with identical student/runtime semantics.

### Linear pairwise baseline limitation

Several feature columns are constant for every node in one frontier:
`frontier_value_max`, `frontier_value_min`, `frontier_value_mean`,
`frontier_value_std`, `beam_width`, and `frontier_size`. In the current pairwise objective,
`x_better - x_worse` makes each of those columns exactly zero. At runtime the same linear
term is also added to every node score, so it cannot change the ordering.

Therefore feature schema v2 does **not** make this independent linear baseline adapt its
ranking rule to K/frontier size or other purely global context. Those columns are retained
in the shared schema for observability and forward compatibility. A context-adaptive model
needs node-by-context interactions, a nonlinear scorer, or a listwise/set-selection model;
that is a later model change rather than an implicit capability of the v2 linear artifact.

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
--seed 0
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
provenance, and metrics. Its `training` metadata also records split seed/fractions,
inverse regularization, split membership, and a `trainer_input_sha256` for every JSONL file.
The hash is always the exact byte stream consumed by `train_stable_pruner.py`. When #54's
one-line ingestion stages normalized Oracle records, the staged path is later rewritten to
the original source-log path while this hash continues to identify the normalized bytes the
trainer actually saw; #54 separately records the original source-log SHA-256 under
`one_line_learning_ingest.source_logs`.

## Offline metrics

Each training split reports:

- pairwise accuracy / weighted pairwise accuracy
- label coverage and selected-`no_target` rates on the full runtime-equivalent frontier
- learned Recall@K against downstream `target_value` top-K
- `ValueTopKPruner` Recall@K on the same known-target population while ranking all nodes
- conditional target-value regret/gap where the selected K and teacher top-K are fully labeled

For a completely separate held-out Oracle collection, evaluate an artifact without
retraining:

```bash
python tools/eval_stable_pruner.py \
  --weights tools/output/stable_pruner_weights.json \
  --log-dir data/combat_oracle_heldout
```

The evaluator reports the same learned-vs-`ValueTopKPruner` ranking/selection metrics and
includes the artifact hash-derived version. Held-out `min_target_gap` and target-source
weights default to the artifact's training metadata; explicit CLI overrides are reported as
such. This makes repeated evaluations of different artifacts unambiguous.

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
- pruner name/version

The report also records the explicit seed list, a SHA-256 of the seed-independent scenario
template, learned pruner identity/version, and the full learned artifact SHA-256 when the
runtime pruner was loaded from an artifact. This is the provenance needed to distinguish
results produced by different board templates or model weights.

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

## Multi-scenario A/B suite

A single Combat state is too narrow for promotion decisions. Use a manifest to run the same
artifact and Beam configuration across multiple scenario templates and seed sets:

```json
{
  "cases": [
    {
      "name": "slime",
      "scenario": "scenarios/slime.json",
      "seeds": [101, 102, 103, 104]
    },
    {
      "name": "cultist",
      "scenario": "scenarios/cultist.json",
      "seeds": [201, 202, 203, 204]
    }
  ]
}
```

Scenario paths are resolved relative to the manifest. Case names must be unique and seeds
inside one case must be unique; invalid manifests fail before starting emulator work.
Run the suite with:

```bash
python -m sts2_training.runner.stable_pruner_ab_suite \
  --manifest data/stable_pruner_ab_suite.json \
  --weights tools/output/stable_pruner_weights.json \
  --search-mode standard \
  --output tools/output/stable_pruner_ab_suite.json
```

Each case retains its complete single-scenario A/B report and provenance. The suite also
flattens all seed pairs into one aggregate summary using the same outcome/search-cost
metrics. `manifest_sha256` fingerprints the canonical case/name/path/seed definition while
each case's `scenario_template_sha256` fingerprints the actual seed-independent board
configuration.

The suite additionally reports `outcome_statistics` over discordant terminal-outcome
pairs: learned wins, baseline wins, learned share among discordant pairs, and a dependency-
free exact two-sided sign-test p-value under a 50/50 null. Ties and unknown outcomes are
excluded from that test. This statistic is diagnostic only; the code intentionally does
not convert it into an automatic promotion threshold or ignore search-cost regressions.

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
