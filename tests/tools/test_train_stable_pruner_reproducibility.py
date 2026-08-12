from __future__ import annotations

import hashlib
from pathlib import Path

import train_stable_pruner

from sts2_training.decision.pruner_features import PRUNER_FEATURE_NAMES


class _Vector(list[float]):
    def tolist(self) -> list[float]:
        return list(self)


class _Scaler:
    scale_ = _Vector([1.0] * len(PRUNER_FEATURE_NAMES))


class _Model:
    coef_ = [_Vector([0.0] * len(PRUNER_FEATURE_NAMES))]


def test_weights_payload_records_split_hyperparameters_and_trainer_input_hashes(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    test = tmp_path / "test.jsonl"
    train.write_bytes(b'{"record":"train"}\n')
    val.write_bytes(b'{"record":"val"}\n')
    test.write_bytes(b'{"record":"test"}\n')
    split = train_stable_pruner.SourceSplit(train=[train], val=[val], test=[test])
    fitted = train_stable_pruner.FittedPairwiseRanker(
        scaler=_Scaler(),
        model=_Model(),
    )

    payload = train_stable_pruner.weights_payload(
        fitted,
        metrics={"train": {}},
        training_files=split.train,
        min_target_gap=1e-6,
        terminal_weight=1.0,
        bootstrap_weight=0.5,
        dataset_files=[train, val, test],
        source_split=split,
        split_seed=7,
        val_fraction=0.2,
        test_fraction=0.2,
        inverse_regularization=3.5,
    )

    training = payload["training"]
    assert training["split"] == {
        "seed": 7,
        "val_fraction": 0.2,
        "test_fraction": 0.2,
    }
    assert training["inverse_regularization"] == 3.5
    manifest = {entry["split"]: entry for entry in training["input_manifest"]}
    assert manifest["train"]["source_path"] == str(train)
    assert manifest["val"]["source_path"] == str(val)
    assert manifest["test"]["source_path"] == str(test)
    assert manifest["train"]["trainer_input_sha256"] == hashlib.sha256(
        train.read_bytes()
    ).hexdigest()
    assert manifest["val"]["trainer_input_sha256"] == hashlib.sha256(
        val.read_bytes()
    ).hexdigest()
    assert manifest["test"]["trainer_input_sha256"] == hashlib.sha256(
        test.read_bytes()
    ).hexdigest()
