# STS2_Training API connection

Training-side implementation for the `sushisk/STS2_RL` API v0.5.

## Unit tests

```bash
python -m pytest tests/api -m "not integration"
```

## Selection audit logging

Pass a `JsonlSelectionLogger` to `TrainingApiClient` to append one flushed UTF-8 JSON
record for each `commit_action` or `emulate_action` call.

```python
from sts2_training.api.client import TrainingApiClient
from sts2_training.selection_log import JsonlSelectionLogger

with JsonlSelectionLogger("logs/selection.jsonl") as selection_log:
    client = TrainingApiClient(
        transport,
        selection_logger=selection_log,
    )
```

Each record contains the public Decision received from RL, the selection request, and the
correlated result. A successful root selection also includes `room_result` when it ends a
room and `run_result` when it ends a Whole Run. Rejected and faulted selections are logged
before their exception is raised.

Only the already-masked DTO received from RL is written. Training does not reconstruct or
add hidden state, and speculative Branch results are never counted as root room/run
results.

## Real Emulator integration test (Windows cmd.exe)

```bat
set STS2_RL_ROOT=C:\path\to\STS2_RL
python -m pytest tests/api/test_api_smoke.py -m integration -vv
```

The combat smoke test covers the complete runtime path without a separate connection
test: Training starts the spawned RL process, `start_instance` loads CoreCLR and the
Emulator and returns a real decision, a Branch Worker executes one speculative action,
and the root action is committed and closed.

## Random root progression integration tests

```bat
set STS2_RL_ROOT=C:\path\to\STS2_RL
python -m pytest tests/api/test_api_random_progression.py -m integration -vv
```

These tests start a fixed independent Combat and a fixed Whole Run, never call
`emulate_action`, and select directly from each published root `legal_actions` list with
a locally seeded PRNG. They verify the configured Combat state, renewed decision IDs,
contiguous root branch logs, at least one non-first random choice, and several distinct
public board states after removing `legal_actions` and DTO-version metadata from the
progress fingerprint.

`LocalProcessTransport` imports `API.api_runtime.RLApiServerProcess` from
`STS2_RL_ROOT`. CLR initialization remains inside the spawned RL child process.
