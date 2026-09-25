"""Prepare and load saved learning stages for read-only playback."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .environment import AdventureEnv
from .learning import Config, TrainingSession

CHECKPOINTS = (1, 50, 100, 200, 500, 1000, 3000, 5000)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "output"


def signature(config):
    """Invalidate old snapshots when rules, map or training settings change."""
    env = AdventureEnv()
    return {
        "version": 1,
        "map": [[tile.name for tile in row] for row in env.layout],
        "rewards": env.rewards,
        "water_passable": True,
        "max_steps": env.max_steps,
        "config": asdict(config),
    }


def train_checkpoints(directory=DEFAULT_OUTPUT, config=Config(), checkpoints=CHECKPOINTS):
    """Train one continuous run and save exactly at each episode boundary."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    session = TrainingSession(config=config)
    for episode in checkpoints:
        session.train(episode - len(session.history))
        np.save(directory / f"q_table_{episode}.npy", session.agent.q, allow_pickle=False)
        # A Q-table alone cannot reconstruct the actual exploratory episode.
        metadata = {
            "signature": signature(config),
            "episode": episode,
            "epsilon": session.epsilon,
            "total_steps": session.total_steps,
            "history": session.history,
            "visits": session.agent.visits.tolist(),
            "steps": session.episode_steps,
        }
        (directory / f"q_table_{episode}.json").write_text(json.dumps(metadata) + "\n")
        print(f"Saved q_table_{episode}.npy", flush=True)


def load_checkpoint(directory, episode, config=Config()):
    """Load a snapshot; arrays are frozen to prevent accidental live learning."""
    directory = Path(directory)
    metadata = json.loads((directory / f"q_table_{episode}.json").read_text())
    if metadata["signature"] != signature(config) or metadata["episode"] != episode:
        raise ValueError("Checkpoint does not match the current training settings.")
    session = TrainingSession(config=config)
    q = np.load(directory / f"q_table_{episode}.npy", allow_pickle=False)
    visits = np.asarray(metadata["visits"], dtype=np.int64)
    if q.shape != session.agent.q.shape or visits.shape != q.shape or not np.isfinite(q).all():
        raise ValueError("Invalid checkpoint array.")
    if len(metadata["history"]) != episode or not metadata["steps"]:
        raise ValueError("Incomplete checkpoint metadata.")
    q.flags.writeable = False
    visits.flags.writeable = False
    session.agent.q = q
    session.agent.visits = visits
    session.history = metadata["history"]
    session.epsilon = metadata["epsilon"]
    session.total_steps = metadata["total_steps"]
    session.episode_steps = metadata["steps"]
    return session


def ensure_checkpoints(directory=DEFAULT_OUTPUT, config=Config()):
    """Reuse complete matching files; otherwise prepare all stages before opening UI."""
    try:
        for episode in CHECKPOINTS:
            load_checkpoint(directory, episode, config)
    except (OSError, ValueError, KeyError, TypeError, EOFError):
        print("Preparing training snapshots before startup...", flush=True)
        train_checkpoints(directory, config)


def recorded_episode(session):
    """The actual training episode, including exploration and failed moves."""
    steps = session.episode_steps
    return {
        "path": [session.env.start] + [step["next_state"] for step in steps],
        "steps": steps,
        "return": sum(step["reward"] for step in steps),
        "success": steps[-1]["terminated"],
        "reason": "goal" if steps[-1]["terminated"] else "step limit",
    }
