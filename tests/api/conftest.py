from __future__ import annotations

import itertools
import os
from pathlib import Path

import pytest

from sts2_training.api.client import TrainingApiClient
from sts2_training.api.local_process_transport import LocalProcessTransport


@pytest.fixture
def rl_root() -> Path:
    value = os.environ.get("STS2_RL_ROOT")
    if not value:
        pytest.skip("STS2_RL_ROOT is not set")
    root = Path(value).expanduser().resolve()
    if not (root / "API" / "api_runtime.py").is_file():
        pytest.fail(f"invalid STS2_RL_ROOT: {root}")
    return root


@pytest.fixture
def local_transport(rl_root: Path):
    transport = LocalProcessTransport(repo_root=rl_root, default_timeout_s=120.0)
    try:
        yield transport
    finally:
        transport.close()


@pytest.fixture
def api_client(local_transport: LocalProcessTransport) -> TrainingApiClient:
    serial = itertools.count(1)
    return TrainingApiClient(
        local_transport,
        request_id_factory=lambda: f"req-integration-{next(serial):06d}",
    )
