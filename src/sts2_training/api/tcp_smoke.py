"""Verify minimal TCP communication with a separately started STS2_RL process."""

from __future__ import annotations

import argparse
import asyncio
import json

from sts2_training.api.asyncio_tcp_transport import AsyncioTcpTransport


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> None:
    async with AsyncioTcpTransport(
        host=args.host,
        port=args.port,
        connect_timeout_s=args.timeout,
    ) as transport:
        response = await transport.ping(timeout_s=args.timeout)
        print(json.dumps(response, ensure_ascii=False, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    asyncio.run(_run(_parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
