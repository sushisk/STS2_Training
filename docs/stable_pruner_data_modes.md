# Stable-pruner data modes

`stable-pruner-learn` separates **what is optimized** (`--learn`) from **how the supplied data is used** (`--data-mode`). Inputs remain the current Training log formats; normalization is automatic.

## `--data-mode auto-split`

Use this for a fresh supervised fit when Training should split the supplied Oracle log files itself:

```bash
stable-pruner-learn data/oracle_logs \
  --learn supervised --start fresh --data-mode auto-split
```

The existing `--val-fraction`, `--test-fraction`, and `--seed` options control the deterministic source-file split. Defaults remain validation 0.1 and test 0.1. The remainder is training data. The artifact records train/val/test metrics.

The shortest fresh command remains equivalent:

```bash
stable-pruner-learn data/oracle_logs
```

because `--data-mode auto` resolves fresh supervised learning to `auto-split`.

## `--data-mode train`

Use this when the supplied logs are already the training partition and must all be used for updates:

```bash
stable-pruner-learn data/train_logs \
  --learn supervised --start fresh --data-mode train \
  --output tools/output/stable_pruner_weights.json
```

For a fresh supervised fit, this forces the trainer's validation and test fractions to zero. For supervised resume and RL resume, `train` retains the existing all-input update behavior.

## `--data-mode validate`

Use this when the supplied Oracle logs are a held-out validation partition:

```bash
stable-pruner-learn data/validation_logs \
  --data-mode validate \
  --weights tools/output/stable_pruner_weights.json \
  --output tools/output/stable_pruner_validation.json
```

Validation performs **no coefficient update**. It invokes the existing held-out Oracle evaluator, checks artifact/teacher provenance with the same default strictness, and writes a JSON validation report. The report retains the original input-log paths, SHA-256 values, record counts, and `data_mode=validate` ingestion metadata.

`validate` currently applies to supervised stable-pruner Oracle evaluation only. RL validation remains a gameplay/A-B evaluation concern rather than replaying trajectory batches as if they were held-out supervised labels.

## Default `--data-mode auto`

The compatibility default resolves as follows:

- fresh supervised -> `auto-split`
- supervised resume -> `train`
- RL resume -> `train`

It never silently turns a resume batch into a train/validation split. If train and validation datasets have already been separated externally, invoke the command once with `--data-mode train` and once with `--data-mode validate`.
