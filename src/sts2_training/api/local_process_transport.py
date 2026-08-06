from __future__ import annotations

import importlib
import os
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from sts2_training.api.transport import (
    JsonObject,
    RuntimeExitedError,
    TransportClosedError,
    TransportError,
)

_ENV_NAME = "STS2_RL_ROOT"


class LocalProcessTransport:
    """Owns one STS2_RL `RLApiServerProcess` and exposes `RlTransport`.

    Importing `API.api_runtime` in the Training process is safe: the RL repository
    initializes pythonnet/CLR only in the spawned child entry point.
    """

    def __init__(
        self,
        *,
        repo_root: Path | str | None = None,
        default_timeout_s: float = 60.0,
    ) -> None:
        if default_timeout_s <= 0:
            raise ValueError("default_timeout_s must be positive")

        self._repo_root = self._resolve_repo_root(repo_root)
        self._ensure_repo_on_sys_path(self._repo_root)
        server_process_class = self._load_server_process_class(self._repo_root)
        self._server_process = server_process_class(
            repo_root=self._repo_root,
            request_timeout_s=default_timeout_s,
        )
        self._default_timeout_s = default_timeout_s
        self._closed = False
        self._call_lock = threading.Lock()

    @property
    def pid(self) -> int | None:
        return getattr(self._server_process, "pid", None)

    @property
    def repo_root(self) -> Path:
        return self._repo_root

    def call(
        self,
        request: Mapping[str, Any],
        *,
        timeout_s: float,
    ) -> JsonObject:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self._closed:
            raise TransportClosedError("transport is closed")
        if not self.is_alive():
            raise RuntimeExitedError("RL runtime process is not alive")

        with self._call_lock:
            previous_timeout = self._server_process.request_timeout_s
            self._server_process.request_timeout_s = timeout_s
            try:
                response = self._server_process.call(dict(request))
            except (EOFError, BrokenPipeError, OSError) as exc:
                if not self.is_alive():
                    raise RuntimeExitedError(
                        "RL runtime process exited during request"
                    ) from exc
                raise TransportError("local process communication failed") from exc
            except RuntimeError as exc:
                if not self.is_alive():
                    raise RuntimeExitedError(
                        "RL runtime process exited during request"
                    ) from exc
                raise TransportError(str(exc)) from exc
            finally:
                self._server_process.request_timeout_s = previous_timeout

        if not isinstance(response, dict):
            raise TransportError("RLApiServerProcess returned a non-dict response")
        return response

    def is_alive(self) -> bool:
        return not self._closed and bool(self._server_process.is_alive())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._server_process.close()

    def __enter__(self) -> "LocalProcessTransport":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _resolve_repo_root(repo_root: Path | str | None) -> Path:
        configured = repo_root if repo_root is not None else os.environ.get(_ENV_NAME)
        if configured is None or not str(configured).strip():
            raise ValueError(
                f"repo_root is required; pass it explicitly or set {_ENV_NAME}"
            )

        resolved = Path(configured).expanduser().resolve()
        runtime_file = resolved / "API" / "api_runtime.py"
        if not runtime_file.is_file():
            raise FileNotFoundError(
                f"STS2_RL API runtime was not found: {runtime_file}"
            )
        return resolved

    @staticmethod
    def _ensure_repo_on_sys_path(repo_root: Path) -> None:
        root_text = str(repo_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

    @classmethod
    def _load_server_process_class(cls, repo_root: Path):
        cls._reject_conflicting_api_package(repo_root)
        module = importlib.import_module("API.api_runtime")
        cls._verify_import_origin(module, repo_root)
        server_process_class = getattr(module, "RLApiServerProcess", None)
        if server_process_class is None:
            raise ImportError("API.api_runtime.RLApiServerProcess was not found")
        return server_process_class

    @staticmethod
    def _reject_conflicting_api_package(repo_root: Path) -> None:
        existing = sys.modules.get("API")
        if existing is None:
            return
        package_file = getattr(existing, "__file__", None)
        package_paths = getattr(existing, "__path__", ())
        candidates = [package_file, *list(package_paths)]
        for candidate in candidates:
            if candidate is None:
                continue
            path = Path(candidate).resolve()
            if path == repo_root / "API" or repo_root in path.parents:
                return
        raise ImportError(
            "a different top-level API package is already imported; start a clean "
            "Training process before creating LocalProcessTransport"
        )

    @staticmethod
    def _verify_import_origin(module: ModuleType, repo_root: Path) -> None:
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise ImportError("API.api_runtime has no import origin")
        resolved = Path(module_file).resolve()
        expected = (repo_root / "API" / "api_runtime.py").resolve()
        if resolved != expected:
            raise ImportError(
                f"API.api_runtime was imported from {resolved}, expected {expected}"
            )
