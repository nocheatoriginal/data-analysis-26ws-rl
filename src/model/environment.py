"""Deterministic grid world using the Gymnasium reset/step interface."""

from collections import deque
from enum import IntEnum

import gymnasium as gym
from gymnasium import spaces


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


ACTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
DEFAULT_MAP = (
    "S...#.....",
    ".##.#.T.#.",
    "..T...~.#.",
    ".###..~...",
    "...T..~.#.",
    ".#.#..~.#.",
    ".#...T...G",
    "...#...#..",
)


class AdventureEnv(gym.Env):
    """One state per tile; actions into water/walls leave the agent in place.

    A trap is traversable and costs -15 on every entry. The only terminal
    tile is the goal. The step limit truncates an episode (not termination).
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 30}
    rewards = {"path": -1.0, "blocked": -5.0, "trap": -15.0, "goal": 100.0}

    def __init__(self, layout=DEFAULT_MAP, max_steps=250, render_mode=None):
        super().__init__()
        self.layout = tuple(layout)
        if (
            not self.layout
            or not self.layout[0]
            or any(len(row) != len(self.layout[0]) for row in self.layout)
        ):
            raise ValueError("Map must be a nonempty rectangle.")
        tiles = "".join(self.layout)
        if set(tiles) - set("S.GT#~") or tiles.count("S") != 1 or tiles.count("G") != 1:
            raise ValueError("Use . T # ~ and exactly one S and G.")
        if max_steps < 1:
            raise ValueError("max_steps must be positive.")
        if render_mode not in (None, "ansi"):
            raise ValueError("Supported render mode: ansi.")
        self.rows, self.cols = len(layout), len(layout[0])
        self.start, self.goal = tiles.index("S"), tiles.index("G")
        self.max_steps, self.render_mode = max_steps, render_mode
        self.observation_space = spaces.Discrete(self.rows * self.cols)
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.state = self.start
        self.steps = 0
        self.finished = False
        if not self.shortest_safe_path(avoid_traps=False):
            raise ValueError("The goal must be reachable from the start.")

    def position(self, state):
        return divmod(int(state), self.cols)

    def tile(self, state):
        row, col = self.position(state)
        return self.layout[row][col]

    def transition(self, state, action):
        """Pure transition function, also useful for inspection and tests."""
        if not self.action_space.contains(action):
            raise ValueError("Action must be an integer from 0 to 3.")
        if state == self.goal:
            return state, 0.0, True, "goal"
        row, col = self.position(state)
        dr, dc = DELTAS[int(action)]
        nr, nc = row + dr, col + dc
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            return state, self.rewards["blocked"], False, "blocked"
        target = nr * self.cols + nc
        tile = self.tile(target)
        if tile in "#~":
            return state, self.rewards["blocked"], False, "blocked"
        if tile == "G":
            event = "goal"
        elif tile == "T":
            event = "trap"
        else:
            event = "path"
        return target, self.rewards[event], tile == "G", event

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state, self.steps, self.finished = self.start, 0, False
        return int(self.state), {"position": self.position(self.state)}

    def step(self, action):
        if self.finished:
            raise RuntimeError("Episode finished. Call reset() before step().")
        self.state, reward, terminated, event = self.transition(self.state, action)
        self.steps += 1
        truncated = self.steps >= self.max_steps and not terminated
        self.finished = terminated or truncated
        return (
            int(self.state),
            reward,
            terminated,
            truncated,
            {
                "position": self.position(self.state),
                "event": event,
            },
        )

    def render(self):
        if self.render_mode == "ansi":
            return "\n".join(
                " ".join(
                    "@" if r * self.cols + c == self.state else tile for c, tile in enumerate(row)
                )
                for r, row in enumerate(self.layout)
            )
        return None

    def shortest_safe_path(self, avoid_traps=True):
        """BFS reference, independent of learning; never used to train the agent."""
        queue = deque([(self.start, [self.start])])
        seen = {self.start}
        while queue:
            state, path = queue.popleft()
            if state == self.goal:
                return path
            for action in Action:
                nxt, _, _, _ = self.transition(state, action)
                if nxt in seen or (avoid_traps and self.tile(nxt) == "T"):
                    continue
                seen.add(nxt)
                queue.append((nxt, path + [nxt]))
        return []
