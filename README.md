# STS2_Training API connection

Training-side implementation for the `sushisk/STS2_RL` API v0.5.

## Test layers

### Unit tests

These tests do not start the RL runtime or load the Emulator.

```bash
python -m pytest tests/api -m "not integration"
```

They cover request construction, response correlation, validation, transport lifecycle,
runtime-exit handling, and branch-operation input rules.

### Emulator connection test

Run this first on the Windows machine that has `STS2_RL` and the Emulator build.

```bat
set STS2_RL_ROOT=C:\path\to\STS2_RL
python -m pytest tests/api/test_emulator_connection.py -m emulator -vv
```

The test verifies the complete connection path:

```text
STS2_Training
  -> LocalProcessTransport
  -> spawned STS2_RL API process
  -> pythonnet/CoreCLR
  -> Sts2Emulator.dll
  -> GameInstance combat reset
  -> first masked decision and legal actions
```

`STS2_RL/Combat/emulator_bridge.py` currently expects the Emulator repository at
`C:\STS2_Emulator` and these build outputs:

```text
C:\STS2_Emulator\Sts2Emulator.Cli\bin\Debug\net8.0\Sts2Emulator.dll
C:\STS2_Emulator\Sts2Emulator.Cli\bin\Debug\net8.0\Sts2Emulator.Cli.runtimeconfig.json
```

Build the Emulator before running the test if either file is missing.

### Branch-worker Emulator smoke test

This additionally verifies that a separate `BranchWorkerPool` process can load its own
Emulator instance and execute one simulated action.

```bat
set STS2_RL_ROOT=C:\path\to\STS2_RL
python -m pytest tests/api/test_emulator_branch_smoke.py -m emulator -vv
```

### Full real-process integration suite

```bat
set STS2_RL_ROOT=C:\path\to\STS2_RL
python -m pytest tests/api -m integration -vv
```

`LocalProcessTransport` imports `API.api_runtime.RLApiServerProcess` from
`STS2_RL_ROOT`. CLR initialization remains inside the spawned RL child process.
Generic CI should run `-m "not integration"` unless the runner has the Windows
Emulator build and `STS2_RL_ROOT` configured.
