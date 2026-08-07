"""Legacy synchronous client compatibility surface.

DTO v0.6 is intentionally async/TCP-first. The old synchronous LocalProcess/RlTransport
path is no longer part of the supported contract because it cannot participate in the
TCP session handshake or server-epoch safety model.
"""

from sts2_training.api.contract import (
    ApiOperationError,
    ApiProtocolError,
    RequestFaultedError,
    RequestRejectedError,
    ROOT_BRANCH_ID,
    ROOT_RNG_ID,
    SCHEMA_VERSION,
)


class TrainingApiClient:
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "DTO v0.6 requires AsyncTrainingApiClient over TcpConnection; "
            "the synchronous TrainingApiClient transport path is retired"
        )


__all__ = [
    "ApiOperationError",
    "ApiProtocolError",
    "RequestFaultedError",
    "RequestRejectedError",
    "ROOT_BRANCH_ID",
    "ROOT_RNG_ID",
    "SCHEMA_VERSION",
    "TrainingApiClient",
]
