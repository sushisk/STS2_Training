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

The same integration file also performs deterministic random root walks for both an
independent combat and a whole run. These tests never call `emulate_action`; they choose
one published legal action at a time with a locally seeded PRNG, commit it directly to
root, and verify that decision IDs, branch logs, and the public board DTO progress for a
minimum number of decisions.

`LocalProcessTransport` imports `API.api_runtime.RLApiServerProcess` from
`STS2_RL_ROOT`. CLR initialization remains inside the spawned RL child process.
