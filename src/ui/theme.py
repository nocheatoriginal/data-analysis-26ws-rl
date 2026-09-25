"""Shared colors and dimensions for a quiet, paper-like teaching interface."""

from ..model.environment import Tile

BACKGROUND = "#f5f4f0"
SURFACE = "#ffffff"
BORDER = "#d6d7d2"
TEXT = "#292e30"
MUTED = "#60686b"
BLUE = "#326b87"
GREEN = "#34694c"
RED = "#a04438"
YELLOW = "#87671c"
BUTTON = "#ffffff"
BUTTON_HOVER = "#e9eeeb"
BUTTON_ACTIVE = "#dce9e1"
SELECTION = "#e4eef2"
VISIT_HIGH = "#accbb6"
TRAIL = "#7e9aa7"

TILE_COLORS = {
    Tile.EMPTY: SURFACE,
    Tile.START: "#e4eee5",
    Tile.GOAL: "#f4ebc9",
    Tile.TRAP: "#f3e1dc",
    Tile.WALL: "#d9dcd9",
    Tile.WATER: "#e0edf2",
}
TILE_NAMES = {
    Tile.EMPTY: "Empty",
    Tile.START: "Start",
    Tile.GOAL: "Goal",
    Tile.TRAP: "Trap",
    Tile.WALL: "Wall",
    Tile.WATER: "Water",
}

DEFAULT_SIZE = (1240, 860)
MIN_SIZE = (1080, 720)
MARGIN = 24
GAP = 24
ROW_HEIGHT = 28
SPEEDS = (1, 5, 10, 20)
VIEWS = ("Policy", "Q-values", "Visits", "Table")
