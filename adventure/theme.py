"""GitHub Dark-inspired palette and shared dimensions, independent of game rules."""

BACKGROUND = "#0d1117"
SURFACE = "#161b22"
BORDER = "#30363d"
TEXT = "#e6edf3"
MUTED = "#a4acb5"
BLUE = "#58a6ff"
GREEN = "#56d364"
RED = "#ff7b72"
YELLOW = "#d29922"
BUTTON = "#21262d"
BUTTON_HOVER = "#30363d"
BUTTON_ACTIVE = "#238636"
SELECTION = "#1c2d41"

TILE_COLORS = {
    ".": BACKGROUND,
    "S": "#16251d",
    "G": "#292416",
    "T": "#2d1b1e",
    "#": "#282e36",
    "~": "#162536",
}
TILE_NAMES = {".": "Path", "S": "Start", "G": "Goal", "T": "Trap", "#": "Wall", "~": "River"}

DEFAULT_SIZE = (1240, 860)
MIN_SIZE = (1080, 760)
MARGIN = 20
GAP = 16
ROW_HEIGHT = 28
SPEEDS = (4, 30, 300, 3000)
VIEWS = ("Policy", "Q-values", "Visits", "Table")

# Light emphasis for small text; less than a normal bold font.
TEXT_STRENGTH = 0.025
