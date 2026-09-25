"""Tabular Q-learning, one inspectable transition at a time."""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .environment import ACTION_NAMES, AdventureEnv


@dataclass(frozen=True)
class Config:
    alpha: float = 0.20
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.997
    seed: int = 7

    def __post_init__(self):
        if not 0 < self.alpha <= 1 or not 0 <= self.gamma < 1:
            raise ValueError("Require 0 < alpha <= 1 and 0 <= gamma < 1.")
        if not 0 <= self.epsilon_min <= self.epsilon_start <= 1:
            raise ValueError("Require 0 <= epsilon_min <= epsilon_start <= 1.")
        if not 0 < self.epsilon_decay <= 1:
            raise ValueError("Require 0 < epsilon_decay <= 1.")


class QLearner:
    """Store one value and one visit count per state-action pair."""

    def __init__(self, states, config=Config()):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.q = np.zeros((states, len(ACTION_NAMES)), dtype=np.float64)
        self.visits = np.zeros_like(self.q, dtype=np.int64)

    def choose(self, state, epsilon):
        """Return an epsilon-greedy action and whether it was exploratory."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(len(ACTION_NAMES))), True
        values = self.q[state]
        # Random tie-breaking prevents an initial bias toward UP.
        candidates = np.flatnonzero(values == values.max())
        return int(self.rng.choice(candidates)), False

    def update(self, state, action, reward, next_state, terminated):
        """Apply the Q-learning equation and return its intermediate values."""
        old = float(self.q[state, action])
        future = 0.0 if terminated else float(self.q[next_state].max())
        target = reward + self.config.gamma * future
        new = old + self.config.alpha * (target - old)
        self.q[state, action] = new
        self.visits[state, action] += 1
        return {
            "state": state,
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "old": old,
            "future": future,
            "target": target,
            "new": new,
            "terminated": terminated,
        }


class TrainingSession:
    """Manage episodes independently of the UI and its animation speed."""

    def __init__(self, env=None, config=Config()):
        self.env = env if env is not None else AdventureEnv()
        self.config = config
        self.agent = QLearner(self.env.observation_space.n, config)
        self.state, _ = self.env.reset(seed=config.seed)
        self.epsilon = config.epsilon_start
        self.history = []
        self.last_update = None
        self.episode_return = 0.0
        self.total_steps = 0
        self.pending_reset = False
        self.trail = [self.state]
        self.episode_steps = []

    def step(self):
        """Learn one transition, leaving its result visible until the next step."""
        if self.pending_reset:
            self.state, _ = self.env.reset()
            self.episode_return = 0.0
            self.trail = [self.state]
            self.episode_steps = []
            self.pending_reset = False
        state = self.state
        action, exploratory = self.agent.choose(state, self.epsilon)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        # Time limits do NOT remove the bootstrap term: only a real terminal does.
        self.last_update = self.agent.update(state, action, reward, next_state, terminated)
        self.last_update.update(exploratory=exploratory, event=info["event"], truncated=truncated)
        self.episode_steps.append(dict(self.last_update))
        self.state = next_state
        self.trail.append(next_state)
        self.episode_return += reward
        self.total_steps += 1
        if terminated or truncated:
            self.finish_episode(success=terminated)
        return self.last_update

    def finish_episode(self, success):
        self.history.append(
            {
                "episode": len(self.history) + 1,
                "return": self.episode_return,
                "steps": self.env.steps,
                "success": success,
                "epsilon": self.epsilon,
            }
        )
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
        self.pending_reset = True

    def train(self, episodes):
        """Complete the requested number of additional episodes."""
        target = len(self.history) + episodes
        while len(self.history) < target:
            self.step()

    def greedy_rollout(self):
        """Evaluate without changing Q-values, training state, or its RNG."""
        state, path, total = self.env.start, [self.env.start], 0.0
        steps = []
        reason = "step limit"
        for _ in range(self.env.max_steps):
            action = int(np.argmax(self.agent.q[state]))
            nxt, reward, terminated, event = self.env.transition(state, action)
            steps.append(
                {
                    "state": state,
                    "action": action,
                    "next_state": nxt,
                    "reward": reward,
                    "event": event,
                    "exploratory": False,
                }
            )
            total += reward
            path.append(nxt)
            if terminated:
                reason = "goal"
                break
            if nxt in path[:-1]:
                reason = "loop"
                break
            state = nxt
        return {
            "path": path,
            "steps": steps,
            "return": total,
            "success": reason == "goal",
            "reason": reason,
        }

    def export(self, directory):
        """Write plain CSV/JSON files for analysis; this is not a checkpoint."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "q_table.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "state",
                    "row",
                    "column",
                    "tile",
                    *ACTION_NAMES,
                    "visits_up",
                    "visits_right",
                    "visits_down",
                    "visits_left",
                ]
            )
            for state, values in enumerate(self.agent.q):
                writer.writerow(
                    [
                        state,
                        *self.env.position(state),
                        self.env.tile(state).name,
                        *values,
                        *self.agent.visits[state],
                    ]
                )
        with (directory / "episodes.csv").open("w", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=["episode", "return", "steps", "success", "epsilon"]
            )
            writer.writeheader()
            writer.writerows(self.history)
        reference = self.env.shortest_safe_path()
        summary = {
            "config": asdict(self.config),
            "map": [[tile.name for tile in row] for row in self.env.layout],
            "rewards": self.env.rewards,
            "max_steps": self.env.max_steps,
            "episodes": len(self.history),
            "transitions": self.total_steps,
            "greedy_evaluation": self.greedy_rollout(),
            "shortest_trap_free_steps": len(reference) - 1 if reference else None,
        }
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return directory
