"""Deterministic grid world using the Gymnasium reset/step interface."""

from collections import deque
from enum import Enum, IntEnum, auto

import gymnasium as gym
from gymnasium import spaces


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


ACTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


class Tile(Enum):
    """The map stores tile types; symbols are only used for display."""

    START = auto()
    WALL = auto()
    EMPTY = auto()
    WATER = auto()
    TRAP = auto()
    GOAL = auto()

    @property
    def symbol(self):
        return {
            Tile.START: "S",
            Tile.WALL: "#",
            Tile.EMPTY: ".",
            Tile.WATER: "~",
            Tile.TRAP: "T",
            Tile.GOAL: "G",
        }[self]


# Keep each map row on one line so the grid remains easy to edit.
# fmt: off
DEFAULT_MAP = (
    (Tile.START, Tile.EMPTY, Tile.EMPTY, Tile.EMPTY, Tile.WALL , Tile.EMPTY, Tile.EMPTY, Tile.EMPTY, Tile.EMPTY, Tile.EMPTY),
    (Tile.EMPTY, Tile.WALL , Tile.WALL , Tile.EMPTY, Tile.WALL , Tile.EMPTY, Tile.TRAP , Tile.EMPTY, Tile.WALL , Tile.EMPTY),
    (Tile.EMPTY, Tile.EMPTY, Tile.TRAP , Tile.EMPTY, Tile.EMPTY, Tile.EMPTY, Tile.WATER, Tile.EMPTY, Tile.WALL , Tile.EMPTY),
    (Tile.EMPTY, Tile.WALL , Tile.WALL , Tile.WALL , Tile.EMPTY, Tile.EMPTY, Tile.WATER, Tile.EMPTY, Tile.EMPTY, Tile.EMPTY),
    (Tile.EMPTY, Tile.EMPTY, Tile.EMPTY, Tile.TRAP , Tile.EMPTY, Tile.EMPTY, Tile.WATER, Tile.EMPTY, Tile.WALL , Tile.EMPTY),
    (Tile.EMPTY, Tile.WALL , Tile.EMPTY, Tile.WALL , Tile.EMPTY, Tile.EMPTY, Tile.WATER, Tile.EMPTY, Tile.WALL , Tile.EMPTY),
    (Tile.EMPTY, Tile.WALL , Tile.EMPTY, Tile.EMPTY, Tile.EMPTY, Tile.TRAP , Tile.EMPTY, Tile.EMPTY, Tile.EMPTY, Tile.GOAL ),
    (Tile.EMPTY, Tile.EMPTY, Tile.EMPTY, Tile.WALL , Tile.EMPTY, Tile.EMPTY, Tile.EMPTY, Tile.WALL , Tile.EMPTY, Tile.EMPTY),
)
# fmt: on


class AdventureEnv(gym.Env):
    """One state per tile; only walls and grid boundaries block movement.

    A trap is traversable and costs -20 on every entry. The only terminal
    tile is the goal. The step limit truncates an episode (not termination).
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 30}
    rewards = {"path": -1.0, "water": -3.0, "blocked": -5.0, "trap": -20.0, "goal": 100.0}

    def __init__(self, layout=DEFAULT_MAP, max_steps=250, render_mode=None):
        super().__init__()
        self.layout = tuple(tuple(row) for row in layout)
        if (
            not self.layout
            or not self.layout[0]
            or any(len(row) != len(self.layout[0]) for row in self.layout)
        ):
            raise ValueError("Map must be a nonempty rectangle.")
        tiles = [tile for row in self.layout for tile in row]
        if any(not isinstance(tile, Tile) for tile in tiles):
            raise ValueError("Map entries must be Tile enum members.")
        if tiles.count(Tile.START) != 1 or tiles.count(Tile.GOAL) != 1:
            raise ValueError("Map must contain exactly one START and one GOAL.")
        if max_steps < 1:
            raise ValueError("max_steps must be positive.")
        if render_mode not in (None, "ansi"):
            raise ValueError("Supported render mode: ansi.")
        self.rows, self.cols = len(self.layout), len(self.layout[0])
        self.start, self.goal = tiles.index(Tile.START), tiles.index(Tile.GOAL)
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
        if tile == Tile.WALL:
            return state, self.rewards["blocked"], False, "blocked"
        if tile == Tile.WATER:
            event = "water"
        elif tile == Tile.GOAL:
            event = "goal"
        elif tile == Tile.TRAP:
            event = "trap"
        else:
            event = "path"
        return target, self.rewards[event], tile == Tile.GOAL, event

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
                    "@" if r * self.cols + c == self.state else tile.symbol
                    for c, tile in enumerate(row)
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
                if nxt in seen or (avoid_traps and self.tile(nxt) == Tile.TRAP):
                    continue
                seen.add(nxt)
                queue.append((nxt, path + [nxt]))
        return []
