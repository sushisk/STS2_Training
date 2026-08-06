# STS2_Training API connection

Training-side implementation for the `sushisk/STS2_RL` API v0.5.

## Unit tests

```bash
python -m pytest tests/api -m "not integration"
```

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
