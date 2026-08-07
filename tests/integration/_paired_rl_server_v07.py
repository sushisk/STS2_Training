from __future__ import annotations

import asyncio
import sys
from pathlib import Path


def _configure_rl_imports(root: Path) -> None:
    for subdirectory in ("Combat", "Run"):
        path = str(root / subdirectory)
        if path not in sys.path:
            sys.path.insert(0, path)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _install_mixed_fault_hook() -> None:
    from API.instance_combat import CombatInstance

    original_finalize = CombatInstance._finalize_branch_result

    def finalize_with_forced_fault(
        self,
        *,
        branch_id,
        parent_branch_id,
        rng_id,
        book,
        branch_log,
        result,
    ):
        if branch_id == "forced-fault":
            return {
                "status": "faulted",
                "branch_id": branch_id,
                "parent_branch_id": parent_branch_id,
                "rng_id": rng_id,
                "error": "forced paired-integration fault",
                "fault_kind": "integration_test",
            }
        return original_finalize(
            self,
            branch_id=branch_id,
            parent_branch_id=parent_branch_id,
            rng_id=rng_id,
            book=book,
            branch_log=branch_log,
            result=result,
        )

    CombatInstance._finalize_branch_result = finalize_with_forced_fault


async def _serve(root: Path, host: str, port: int) -> None:
    _configure_rl_imports(root)
    _install_mixed_fault_hook()

    from API.server import RLApiServer
    from API.tcp_server import AsyncioTcpServer

    dispatcher = RLApiServer()
    server = AsyncioTcpServer(
        dispatcher.handle_request,
        server_epoch=dispatcher.server_epoch,
        host=host,
        port=port,
    )
    await server.start()
    print("PAIRED_RL_READY", flush=True)
    try:
        await server.serve_forever()
    finally:
        await server.close()
        dispatcher.close_all()


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: _paired_rl_server_v07.py RL_ROOT HOST PORT")
    root = Path(sys.argv[1]).resolve()
    asyncio.run(_serve(root, sys.argv[2], int(sys.argv[3])))


if __name__ == "__main__":
    main()
