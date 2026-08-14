"""Capture terminal DTOs for RL reward calculation."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any

class TerminalDtoCapture:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.final_dto: dict[str, Any] | None = None
        self.terminal_dtos: list[dict[str, Any]] = []
    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)
    async def get_decision(self, *args: Any, **kwargs: Any) -> Any:
        return self._capture(await self.inner.get_decision(*args, **kwargs))
    async def commit_action(self, *args: Any, **kwargs: Any) -> Any:
        return self._capture(await self.inner.commit_action(*args, **kwargs))
    def _capture(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            dto = value.get("masked_emulator_dto")
            if isinstance(dto, Mapping) and (dto.get("terminal") is True or dto.get("run_terminal") is True):
                self.final_dto = dict(dto)
                self.terminal_dtos.append(self.final_dto)
        return value
