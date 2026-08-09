from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sts2_training.runner.episode import EpisodeResult
from sts2_training.runner.self_play import run_self_play_batch


class _Connection:
    client_session_id = "session-test"

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass


class SelfPlayReproducibilityTest(unittest.IsolatedAsyncioTestCase):
    async def test_generated_seed_is_recorded_and_seeds_fallback_policy(self) -> None:
        episode = EpisodeResult(
            instance_id="instance-1",
            decisions_made=0,
            final_dto={"terminal": True, "outcome": "victory"},
            elapsed_s=0.0,
        )
        fake_engine = object()

        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch("sts2_training.runner.self_play.random.randint", return_value=123456),
                mock.patch(
                    "sts2_training.runner.self_play.HeuristicCombatSelector"
                ) as selector_cls,
                mock.patch(
                    "sts2_training.runner.self_play.CombatDecisionEngine",
                    return_value=fake_engine,
                ) as engine_cls,
                mock.patch(
                    "sts2_training.runner.self_play.start_new_run",
                    new=mock.AsyncMock(return_value=episode),
                ) as start_new_run,
            ):
                results = await run_self_play_batch(
                    character_id="IRONCLAD",
                    num_runs=1,
                    connection_factory=_Connection,
                    output_dir=Path(tmp),
                )

        self.assertEqual(len(results), 1)
        self.assertIn("-seed-123456-", results[0].run_id)
        start_new_run.assert_awaited_once()
        self.assertEqual(start_new_run.await_args.kwargs["seed"], 123456)
        self.assertIs(start_new_run.await_args.kwargs["engine"], fake_engine)

        selector_cls.assert_called_once()
        seeded_rng = selector_cls.call_args.args[0]
        self.assertIsInstance(seeded_rng, random.Random)
        self.assertEqual(seeded_rng.random(), random.Random(123456).random())
        self.assertIs(engine_cls.call_args.kwargs["fallback_selector"], selector_cls.return_value)


if __name__ == "__main__":
    unittest.main()
