# Oracle Teacher Provenance Contract

PR #47 makes Oracle labels depend on an isolated, identifiable teacher stack. The
supervised stable-pruner pipeline therefore treats teacher provenance as part of the label
contract rather than incidental logging.

## Oracle input contract

Stable-pruner training/evaluation consumes only `combat_oracle_decision` records using the
current Oracle JSONL schema (v3). Each record must carry the provenance emitted by the
Oracle collector, including:

- teacher Policy / inner Policy / coverage Policy class identity
- teacher ValueModel class identity
- `teacher_policy_metadata`
- `teacher_inner_policy_metadata`
- `teacher_value_metadata`
- Oracle pruner name/version and RNG sampling mode
- collection `training_commit` when available

The metadata maps come from the `PolicyModel.oracle_provenance()` /
`ValueModel.oracle_provenance()` seam introduced by PR #47. Learned teachers should include
checkpoint/version/config hashes or equivalent fields sufficient to identify the label
generator.

The stable-pruner loader canonicalizes the complete provenance object and records its
SHA-256 fingerprint. JSON object key order does not affect the fingerprint.

## Default: one teacher configuration

`load_pruner_frontiers()` rejects a dataset containing more than one teacher provenance
fingerprint by default. This prevents silently pooling labels from different Value weights,
Policy checkpoints, or other teacher configurations.

`tools/train_stable_pruner.py` uses the same strict default. Intentional teacher mixtures
require:

```bash
python tools/train_stable_pruner.py \
  --log-dir data/combat_oracle \
  --output tools/output/stable_pruner_weights.json \
  --allow-mixed-teachers
```

This flag changes only the consistency rule; it does not erase provenance.

## Artifact provenance

A newly trained artifact records two provenance summaries:

- `oracle_teacher_provenance`: provenance for the files actually assigned to the training
  split.
- `oracle_dataset_provenance`: provenance for all Oracle JSONL files submitted before the
  train/validation/test split.

Each summary includes the Oracle schema version, teacher fingerprints, record/source-file
counts, and the complete provenance payload for every teacher fingerprint. This lets an
artifact be traced back to the exact Oracle label-generator configuration even when mixed
teachers were explicitly allowed.

These fields are additive artifact metadata; runtime linear inference remains dependency
free.

## Held-out evaluation

`tools/eval_stable_pruner.py` validates held-out Oracle provenance before computing ranking
metrics. By default:

1. the held-out dataset itself must contain one teacher configuration;
2. the learned-pruner artifact must contain training teacher provenance; and
3. the held-out teacher fingerprint set must equal the artifact training-teacher set.

Intentional exceptions are explicit:

```text
--allow-mixed-teachers   permit multiple teacher fingerprints in held-out Oracle data
--allow-teacher-mismatch evaluate a teacher set different from the artifact's training set
```

When mismatch is explicitly allowed, the report sets `teacher_provenance_match` to false
and still emits both `training_oracle_teacher_provenance` and
`evaluation_oracle_teacher_provenance`. A mismatch is therefore visible rather than being
silently converted into an ordinary held-out score.

## Why this is strict

Oracle `target_value` is a property of the teacher/search configuration that produced it.
Changing a Value checkpoint or Policy configuration can change label ordering even when
state/action features are identical. Treating provenance as a dataset contract avoids
training on an unmarked mixture of different objectives and makes later A/B or RL results
auditable.