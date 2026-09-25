"""Check exact training boundaries, recorded errors, and reuse of saved stages."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.model.checkpoints import (
    ensure_checkpoints,
    load_checkpoint,
    recorded_episode,
    train_checkpoints,
)
from src.model.learning import Config, TrainingSession


class CheckpointTests(unittest.TestCase):
    def test_saved_tables_and_recordings_match_continuous_training(self):
        with tempfile.TemporaryDirectory() as directory:
            train_checkpoints(directory, checkpoints=(1, 50))
            reference = TrainingSession()
            for episode in (1, 50):
                reference.train(episode - len(reference.history))
                loaded = load_checkpoint(directory, episode)
                np.testing.assert_array_equal(loaded.agent.q, reference.agent.q)
                np.testing.assert_array_equal(loaded.agent.visits, reference.agent.visits)
                self.assertEqual(loaded.history, reference.history)
                recording = recorded_episode(loaded)
                self.assertEqual(recording["steps"], reference.episode_steps)
                self.assertEqual(recording["return"], reference.history[-1]["return"])
                for step in recording["steps"]:
                    nxt, reward, terminal, event = loaded.env.transition(
                        step["state"], step["action"]
                    )
                    self.assertEqual(
                        (nxt, reward, terminal, event),
                        (step["next_state"], step["reward"], step["terminated"], step["event"]),
                    )
                if episode == 1:
                    self.assertTrue(any(step["event"] == "blocked" for step in recording["steps"]))
                with self.assertRaises(ValueError):
                    loaded.agent.q[0, 0] = 99
            with self.assertRaises(ValueError):
                load_checkpoint(directory, 1, Config(seed=42))
            np.save(Path(directory) / "q_table_1.npy", np.zeros((2, 4)))
            with self.assertRaises(ValueError):
                load_checkpoint(directory, 1)

    def test_existing_files_are_reused_and_missing_files_regenerated(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.model.checkpoints.CHECKPOINTS", (1, 50)):
                train_checkpoints(directory, checkpoints=(1, 50))
                with patch("src.model.checkpoints.train_checkpoints") as train:
                    ensure_checkpoints(directory)
                    train.assert_not_called()
                    (Path(directory) / "q_table_50.npy").unlink()
                    ensure_checkpoints(directory)
                    train.assert_called_once()
