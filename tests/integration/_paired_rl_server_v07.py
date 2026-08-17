from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

_SERVER_MAX_MESSAGE_BYTES = 4096
_PAIRED_MAX_EMULATE_ACTIONS_ITEMS = 16


def _configure_rl_imports(root: Path) -> None:
    for subdirectory in ("Combat", "Run"):
        path = str(root / subdirectory)
        if path not in sys.path:
            sys.path.insert(0, path)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _install_mixed_fault_hook() -> None:
    """Force one item to fail inside the real worker execution path.

    The previous hook rewrote a successful result during response finalization, which
    only tested Training's mixed-status parsing. This hook corrupts one admitted
    candidate before it is serialized to a Branch Worker so the worker itself returns a
    normalized fault. Finalization only relabels that already-real worker fault with the
    stable integration-test fault kind expected by the paired assertion.
    """

    from API.instance_combat import CombatInstance
    from search.decision_context import SemanticAction

    original_validate = CombatInstance._validate_emulate_actions_item
    original_finalize = CombatInstance._finalize_branch_result

    def validate_with_forced_worker_fault(self, item):
        admitted = original_validate(self, item)
        if admitted.branch_id != "forced-fault":
            return admitted
        broken_candidate = replace(
            admitted.candidate,
            semantic_action=SemanticAction("__paired_forced_worker_fault__", ""),
        )
        return replace(admitted, candidate=broken_candidate)

    def finalize_with_verified_worker_fault(
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
            if result.status == "success":
                raise AssertionError("forced paired fault unexpectedly succeeded in worker")
            diagnostics = dict(result.diagnostics or {})
            diagnostics["fault_kind"] = "integration_test"
            diagnostics.setdefault("message", "forced paired worker fault")
            result = replace(result, diagnostics=diagnostics)
        return original_finalize(
            self,
            branch_id=branch_id,
            parent_branch_id=parent_branch_id,
            rng_id=rng_id,
            book=book,
            branch_log=branch_log,
            result=result,
        )

    CombatInstance._validate_emulate_actions_item = validate_with_forced_worker_fault
    CombatInstance._finalize_branch_result = finalize_with_verified_worker_fault


async def _serve(root: Path, host: str, port: int) -> None:
    _configure_rl_imports(root)
    _install_mixed_fault_hook()

    from API.server import RLApiServer
    from API.tcp_server import AsyncioTcpServer

    # Use a non-default capacity so the paired test proves Training discovers the
    # configured RL value instead of accidentally depending on Combat's default 64.
    dispatcher = RLApiServer(
        instance_factory_kwargs={
            "combat": {"max_branches": _PAIRED_MAX_EMULATE_ACTIONS_ITEMS}
        }
    )
    server = AsyncioTcpServer(
        dispatcher.handle_request,
        server_epoch=dispatcher.server_epoch,
        host=host,
        port=port,
        max_message_bytes=_SERVER_MAX_MESSAGE_BYTES,
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
