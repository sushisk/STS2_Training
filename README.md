# STS2_Training API connection

Training-side implementation for the `sushisk/STS2_RL` API v0.5.

## Unit tests

```bash
python -m pytest tests/api -m "not integration"
```

## Real process integration test (Windows cmd.exe)

```bat
set STS2_RL_ROOT=C:\path\to\STS2_RL
python -m pytest tests/api/test_api_smoke.py -m integration
```

`LocalProcessTransport` imports `API.api_runtime.RLApiServerProcess` from
`STS2_RL_ROOT`. CLR initialization remains inside the spawned RL child process.
