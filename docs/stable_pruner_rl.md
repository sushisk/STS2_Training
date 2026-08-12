# Stable Frontier Pruner RL Fine-Tuning

This stage starts only after the supervised stable-pruner baseline from PR #50 exists.
It keeps the search-control surface narrow: only stable/resolved frontier survivor choice is
explored. Policy candidate generation, continuation handling, beam width, stopping, and Whole
Run active-branch capacity remain unchanged.

## Why the first RL policy is stochastic top-K

The supervised artifact already provides one linear score per stable node. RL therefore does
not learn pruning from scratch. `PlackettLuceLinearStableFrontierPruner` wraps that scorer and
samples an ordered top-K choice without replacement.

The sampled order is used only to define an exact behavior-policy probability for REINFORCE.
The Beam receives the sampled survivor *set* in deterministic base-score order, so exploration
does not additionally randomize parent expansion order.

When `frontier_size <= beam_width` every node survives. No stochastic choice exists, so the
wrapper preserves the supervised ordering and emits no RL step.

## On-policy trajectory collection

Use the real emulator and the existing fixed-seed A/B contract:

```bash
python -m sts2_training.runner.stable_pruner_rl \
  --scenario data/scenarios/slime.json \
  --weights tools/output/stable_pruner_weights.json \
  --seeds 101,102,103,104 \
  --temperature 1.0 \
  --output tools/output/stable_pruner_rl_batch.jsonl
```

For each seed:

1. the baseline arm uses `ValueTopKPruner`;
2. the learned arm uses a fresh stochastic wrapper around the exact input artifact;
3. arm order alternates across seeds;
4. only the learned arm's stochastic stable-prune decisions are recorded;
5. the paired terminal-outcome and search-cost delta becomes the episode reward.

The default reward is only the paired terminal outcome delta. Optional compute penalties are:

```text
--node-cost-weight
--beam-ms-cost-weight
```

The reward is:

```text
(learned_outcome - baseline_outcome)
- node_cost_weight * (learned_nodes - baseline_nodes)
- beam_ms_cost_weight * (learned_beam_ms - baseline_beam_ms)
```

Victory/Win maps to 1 and Defeat/Loss maps to 0. Unknown terminal outcomes are not training
examples. Episodes with no actual stable-pruning choice are also skipped.

## Trajectory contract

Each JSONL record stores:

- exact behavior artifact SHA-256;
- stochastic pruner name/version;
- temperature and sampler seed;
- scenario-template SHA-256 and Beam configuration;
- paired baseline/learned outcomes and search-cost deltas;
- scalar reward and its components;
- every stochastic prune step.

Each prune step stores the complete stable-pruner feature matrix, behavior scores, sampled
Plackett-Luce indices, deterministic returned survivor order, and exact selection log
probability. This is enough to audit and recompute the policy-gradient term without replaying
the emulator.

The stochastic collector deliberately records only the feature-domain already used by the
supervised pruner. Opaque decision-local `action_id` values are not learned as identities.

## One update per batch

Apply exactly one policy-gradient update to the artifact that generated the batch:

```bash
python tools/update_stable_pruner_rl.py \
  --weights tools/output/stable_pruner_weights.json \
  --trajectory tools/output/stable_pruner_rl_batch.jsonl \
  --output tools/output/stable_pruner_rl_weights.json \
  --learning-rate 0.01 \
  --gradient-clip-norm 5.0
```

The updater fails if a trajectory's behavior artifact SHA does not exactly match the input
artifact. It also recomputes every logged behavior score and Plackett-Luce log probability
from the artifact and rejects inconsistent records.

The update is the episode-level REINFORCE estimator:

```text
gradient = mean_episode(
    paired_reward * sum_prune_step(grad log pi(sampled_top_k | frontier))
)
```

The exact Plackett-Luce score gradient is used. Gradient norm clipping limits one update's
size.

The tool intentionally has no `--epochs` option. Reusing the same behavior batch for multiple
policy changes would no longer be a clean on-policy update. The intended loop is:

```text
supervised artifact
    -> collect stochastic paired trajectories
    -> one REINFORCE update
    -> new artifact
    -> collect a fresh batch with the new artifact
    -> repeat
```

## Artifact provenance

The updated JSON remains loadable by `LinearStableFrontierPruner`; runtime inference therefore
remains dependency-free. Existing supervised/Oracle provenance is preserved.

Each RL update appends `rl_finetuning_history` containing:

- parent artifact SHA-256;
- trajectory files;
- learning rate and gradient clipping;
- update commit when available;
- reward/example counts and gradient norms.

The next collection batch must point at the newly produced artifact SHA, which makes the
on-policy chain auditable.

## Promotion boundary

This stage does not automatically promote a learned pruner. Keep the existing held-out Oracle
metrics and fixed-seed real-emulator evaluation as separate diagnostics.

RL should be considered an improvement only when the new artifact is competitive on the
teacher-distillation checks and improves paired gameplay/search utility on held-out scenario
and seed sets. A single positive training-batch reward is not a promotion criterion.

## Still out of scope

This first RL stage does not learn:

- dynamic beam width;
- parent expansion scheduling;
- continuation allocation;
- stopping;
- PolicyModel action proposal;
- Whole Run branch-capacity allocation;
- arbitrary subset/set-network selection.

Those belong to the later `SearchController` stage after stable-frontier RL is measurable and
reproducible.
