from __future__ import annotations

import unittest

from sts2_training.decision.beam_search import BeamSearchConfig, BeamSearchEngine
from sts2_training.decision.oracle_search import BudgetedOracleCollector, OracleCollectionConfig
from sts2_training.decision.policy import ActionCandidate, PolicyModel
from sts2_training.decision.value import ValueModel


class _SingleActionPolicy(PolicyModel):
    def propose(self, legal_actions, masked_emulator_dto, *, top_k):
        del masked_emulator_dto
        available = [
            action
            for action in legal_actions
            if action.get("is_available") is not False
        ]
        return [
            ActionCandidate(action_id=str(action["action_id"]))
            for action in available[:top_k]
        ]


class _ZeroValue(ValueModel):
    def evaluate_batch(self, masked_emulator_dtos):
        return [0.0 for _dto in masked_emulator_dtos]


class _RecordingClient:
    def __init__(self) -> None:
        self.rng_ids: list[int] = []
        self.branch_ids: list[str] = []

    async def emulate_actions(
        self,
        instance_id,
        items,
        *,
        timeout_s,
        simulation_options=None,
    ):
        del instance_id, timeout_s, simulation_options
        branch_results = {}
        for item in items:
            rng_id = item["rng_id"]
            branch_id = item["branch_id"]
            self.rng_ids.append(rng_id)
            self.branch_ids.append(branch_id)
            branch_results[branch_id] = {
                "status": "completed",
                "decision_point_id": f"terminal-{branch_id}",
                "masked_emulator_dto": {
                    "terminal": True,
                    "outcome": "victory",
                    "legal_actions": [],
                },
            }
        return {"branch_results": branch_results}

    async def cancel_branches(self, instance_id, branch_ids, *, timeout_s):
        del instance_id, branch_ids, timeout_s
        return {}

    async def release_branches(self, instance_id, branch_ids, *, timeout_s):
        del instance_id, branch_ids, timeout_s
        return {}


class OracleRuntimeRngIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_oracle_runtime_oracle_searches_use_distinct_wire_rng_ids(self) -> None:
        client = _RecordingClient()
        beam_config = BeamSearchConfig(
            beam_width=1,
            top_k_actions=1,
            max_depth=1,
        )
        runtime = BeamSearchEngine(
            client,
            policy=_SingleActionPolicy(),
            value_fn=_ZeroValue(),
            config=beam_config,
        )
        oracle = BudgetedOracleCollector.from_beam_engine(
            runtime,
            config=OracleCollectionConfig(
                beam_config=beam_config,
                target_beam_width=1,
                exhaustive_root_actions=True,
            ),
        )
        root = {
            "decision_point_id": "root-decision",
            "masked_emulator_dto": {
                "legal_actions": [
                    {
                        "action_id": "play",
                        "action_type": "card",
                        "is_available": True,
                    }
                ]
            },
        }

        await oracle.collect("instance", root, timeout_s=1.0)
        await runtime.search("instance", root, timeout_s=1.0)
        await oracle.collect("instance", root, timeout_s=1.0)

        self.assertEqual(client.rng_ids, [1, 2, 3])
        self.assertEqual(len(set(client.rng_ids)), 3)
        self.assertEqual(len(set(client.branch_ids)), 3)


if __name__ == "__main__":
    unittest.main()
