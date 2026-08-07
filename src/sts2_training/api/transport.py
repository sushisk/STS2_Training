from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class RetryRequest:
    """Exact serialized request for replaying one unresolved session sequence."""

    serialized_payload: str

    @classmethod
    def from_message(cls, message: Mapping[str, Any]) -> "RetryRequest":
        return cls(json.dumps(dict(message), ensure_ascii=False, separators=(",", ":")))

    def to_message(self) -> JsonObject:
        value = json.loads(self.serialized_payload)
        if not isinstance(value, dict):
            raise ValueError("retry request must contain a JSON object")
        return value

    @property
    def request_id(self) -> str | None:
        value = self.to_message().get("request_id")
        return value if isinstance(value, str) else None

    @property
    def operation(self) -> str | None:
        value = self.to_message().get("operation")
        return value if isinstance(value, str) else None

    @property
    def request_seq(self) -> int | None:
        value = self.to_message().get("request_seq")
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @property
    def client_session_id(self) -> str | None:
        value = self.to_message().get("client_session_id")
        return value if isinstance(value, str) else None


class TransportError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        completion_uncertain: bool = False,
        retry_request: RetryRequest | None = None,
    ) -> None:
        super().__init__(message)
        self.completion_uncertain = completion_uncertain
        self.retry_request = retry_request


class ServerEpochChangedError(TransportError):
    """RL restarted; the logical client session cannot be continued safely."""

    def __init__(
        self,
        *,
        expected_epoch: str,
        actual_epoch: str,
        completion_uncertain: bool = False,
        retry_request: RetryRequest | None = None,
    ) -> None:
        super().__init__(
            f"RL server epoch changed from {expected_epoch!r} to {actual_epoch!r}",
            completion_uncertain=completion_uncertain,
            retry_request=retry_request,
        )
        self.expected_epoch = expected_epoch
        self.actual_epoch = actual_epoch


class TransportClosedError(TransportError):
    pass


class RuntimeExitedError(TransportError):
    pass
