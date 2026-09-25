"""Shared colors and dimensions for a quiet, paper-like teaching interface."""

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
    ".": SURFACE,
    "S": "#e4eee5",
    "G": "#f4ebc9",
    "T": "#f3e1dc",
    "#": "#d9dcd9",
    "~": "#e0edf2",
}
TILE_NAMES = {".": "Path", "S": "Start", "G": "Goal", "T": "Trap", "#": "Wall", "~": "River"}

DEFAULT_SIZE = (1240, 860)
MIN_SIZE = (1080, 720)
MARGIN = 24
GAP = 24
ROW_HEIGHT = 28
SPEEDS = (4, 30, 300, 3000)
VIEWS = ("Policy", "Q-values", "Visits", "Table")
