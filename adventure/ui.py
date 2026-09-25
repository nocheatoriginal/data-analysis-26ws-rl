"""Pygame UI: layout, drawing, input handling and a timed training loop.

Layout and clicks use logical window coordinates. display.py renders text at
the physical pixel density of the current display, including Retina screens.
"""

import time
from pathlib import Path

import numpy as np
import pygame as pg

from . import theme
from .display import Display, TextLabel
from .environment import ACTION_NAMES, DELTAS, AdventureEnv
from .learning import TrainingSession


class Dashboard:
    def __init__(self, session: TrainingSession, output="outputs", diagnose_display=False):
        pg.init()
        desktop_width, desktop_height = pg.display.get_desktop_sizes()[0]
        size = (
            max(theme.MIN_SIZE[0], min(theme.DEFAULT_SIZE[0], desktop_width - 80)),
            max(theme.MIN_SIZE[1], min(theme.DEFAULT_SIZE[1], desktop_height - 100)),
        )
        self.display = Display(size)
        self.window = self.display.surface
        self.buttons = []
        self.session = session
        self.output = Path(output)
        self.diagnose_display = diagnose_display
        self.next_diagnostic = 0.0

        self.alive = True
        self.running = False
        self.target = None  # Episode at which accelerated training should pause.
        self.speed = 1
        self.credit = 0.0  # Accumulated time expressed as training steps.
        self.view = 0
        self.detail = "chart"  # Show one explanation at a time.
        self.selected = session.env.start
        self.scroll = 0
        self.watch = None
        self.watch_index = 0
        self.watch_elapsed = 0.0
        self.status = "Ready. Run training or press N to learn one transition."
        if session.history:
            self.status = (
                f"{len(session.history):,} episodes trained. Press W to watch the learned route."
            )
        self.update_layout()

    def update_layout(self):
        """Share the same rectangles between drawing and mouse hit detection."""
        width, height = self.window.get_size()
        margin, gap = theme.MARGIN, theme.GAP
        left_width = (width - 2 * margin - gap) * 55 // 100
        right_x = margin + left_width + gap
        right_width = width - right_x - margin
        body_bottom = height - 122

        self.world_panel = pg.Rect(margin, 150, left_width, body_bottom - 150)
        self.detail_panel = pg.Rect(right_x, 198, right_width, body_bottom - 198)
        self.inspector_panel = self.detail_panel
        self.update_panel = self.detail_panel
        self.chart_panel = self.detail_panel
        self.legend_panel = pg.Rect(margin, body_bottom + 12, left_width, 78)
        self.settings_panel = pg.Rect(right_x, body_bottom + 12, right_width, 78)

        env = self.session.env
        self.cell_size = min(
            (left_width - 42) // env.cols, (self.world_panel.height - 76) // env.rows
        )
        board_width = self.cell_size * env.cols
        board_height = self.cell_size * env.rows
        self.board = pg.Rect(
            margin + (left_width - board_width) // 2,
            self.world_panel.y + 54,
            board_width,
            board_height,
        )
        self.table_body = pg.Rect(margin + 12, 222, left_width - 24, self.world_panel.height - 108)
        self.visible_rows = self.table_body.height // theme.ROW_HEIGHT
        self.table_body.height = self.visible_rows * theme.ROW_HEIGHT
        self.clamp_scroll()

    def clamp_scroll(self):
        last_page = max(0, len(self.session.agent.q) - self.visible_rows)
        self.scroll = max(0, min(last_page, self.scroll))

    def font(self, size=14, bold=False, mono=False):
        return self.display.font(size, bold, mono)

    def text(
        self,
        value,
        x,
        y,
        size=14,
        color=theme.TEXT,
        bold=False,
        mono=False,
        center=False,
        max_width=None,
    ):
        """Measure in window coordinates; defer rasterization until presentation."""
        font = self.font(size, bold, mono)
        label = str(value)
        if max_width is not None and font.size(label)[0] > max_width:
            while label and font.size(label + "...")[0] > max_width:
                label = label[:-1]
            label += "..."
        self.display.labels.append(TextLabel(label, (x, y), size, color, bold, mono, center))
        return font.size(label)[0]

    def panel(self, rect, title):
        pg.draw.rect(self.window, theme.SURFACE, rect, border_radius=4)
        self.text(title, rect.x + 12, rect.y + 12, bold=True)

    def button(self, label, x, action, active=False, y=100):
        width = self.font().size(label)[0] + 24
        rect = pg.Rect(x, y, width, 34)
        color = theme.BUTTON_ACTIVE if active else theme.BUTTON
        if not active and rect.collidepoint(pg.mouse.get_pos()):
            color = theme.BUTTON_HOVER
        pg.draw.rect(self.window, color, rect, border_radius=4)
        pg.draw.rect(self.window, theme.BORDER, rect, 1, border_radius=4)
        self.text(label, *rect.center, center=True)
        self.buttons.append((rect, action))
        return rect.right + 8

    @staticmethod
    def value_color(value):
        if value > 0:
            return theme.GREEN
        if value < 0:
            return theme.RED
        return theme.MUTED

    def coords(self, state):
        row, column = self.session.env.position(state)
        return (
            self.board.x + column * self.cell_size + self.cell_size // 2,
            self.board.y + row * self.cell_size + self.cell_size // 2,
        )

    def arrow(self, center, action, color=theme.MUTED):
        x, y = center
        dy, dx = DELTAS[action]
        tip = (x + dx * 9, y + dy * 9)
        pg.draw.line(self.window, color, (x - dx * 7, y - dy * 7), tip, 2)
        for side in (-1, 1):
            wing = (tip[0] - dx * 5 + dy * side * 5, tip[1] - dy * 5 - dx * side * 5)
            pg.draw.line(self.window, color, tip, wing, 2)

    def draw_header(self):
        self.text("relearn", theme.MARGIN, 20, 26, bold=True)
        self.text("Learn by trying. Find a route to the goal.", theme.MARGIN, 57, color=theme.MUTED)
        recent = self.session.history[-100:]
        success = f"{sum(item['success'] for item in recent) / len(recent):.0%}" if recent else "--"
        stats = (
            ("Episodes", f"{len(self.session.history):,}"),
            ("Exploration", f"{self.session.epsilon:.1%}"),
            ("Success (last 100)", success),
        )
        for index, (label, value) in enumerate(stats):
            x = self.inspector_panel.x + index * self.inspector_panel.width // 3
            self.text(label, x, 22, color=theme.MUTED)
            self.text(value, x, 47, 20, mono=True)
        controls = (
            ("Pause" if self.running else "Run", "run", self.running),
            ("Step", "step", False),
            ("Train +500", "batch", self.target is not None),
            ("Watch route", "watch", self.watch is not None),
            (f"View: {theme.VIEWS[self.view]}", "view", False),
            (f"{theme.SPEEDS[self.speed]:,} steps/s", "speed", False),
            ("Export", "export", False),
            ("Reset", "reset", False),
        )
        x = theme.MARGIN
        for label, action, active in controls:
            x = self.button(label, x, action, active)

    def draw_q_values(self, state, rect):
        pg.draw.line(self.window, theme.BORDER, rect.topleft, rect.bottomright)
        pg.draw.line(self.window, theme.BORDER, rect.topright, rect.bottomleft)
        labels = [str(round(value)) for value in self.session.agent.q[state]]
        # Left and right values each get half a cell, with a gap between them.
        size = 13
        available_width = self.cell_size // 2 - 4
        while (
            size > 9
            and max(self.font(size, mono=True).size(label)[0] for label in labels) > available_width
        ):
            size -= 1
        for action, (dy, dx) in enumerate(DELTAS):
            self.text(
                labels[action],
                rect.centerx + dx * (self.cell_size // 4),
                rect.centery + dy * (self.cell_size // 3),
                size,
                self.value_color(self.session.agent.q[state, action]),
                mono=True,
                center=True,
            )

    def draw_tile(self, state, max_visits):
        tile = self.session.env.tile(state)
        center = self.coords(state)
        rect = pg.Rect(0, 0, self.cell_size, self.cell_size)
        rect.center = center
        visits = int(self.session.agent.visits[state].sum())
        color = pg.Color(theme.TILE_COLORS[tile])
        if self.view == 2 and tile not in "#~G":
            intensity = np.log1p(visits) / np.log1p(max_visits)
            color = pg.Color(theme.SURFACE).lerp(pg.Color(theme.VISIT_HIGH), intensity)
        pg.draw.rect(self.window, color, rect)
        pg.draw.rect(self.window, theme.BORDER, rect, 1)

        if tile in "#~G":
            symbol_color = {"#": theme.MUTED, "~": theme.BLUE, "G": theme.YELLOW}[tile]
            self.text(tile, *center, 20, symbol_color, mono=True, center=True)
        elif self.view == 1:
            self.draw_q_values(state, rect)
        elif self.view == 2:
            self.text(visits, *center, 16, mono=True, center=True)
        elif visits:
            action = int(np.argmax(self.session.agent.q[state]))
            self.arrow(center, action)
        else:
            pg.draw.circle(self.window, theme.BORDER, center, 2)

        if tile in "ST" and self.view != 1:
            label_color = theme.GREEN if tile == "S" else theme.RED
            self.text(tile, rect.x + 5, rect.y + 3, 13, label_color, bold=True)
        if state == self.selected:
            pg.draw.rect(self.window, theme.BLUE, rect.inflate(-2, -2), 2)

    def draw_board(self):
        self.panel(self.world_panel, f"Environment · {theme.VIEWS[self.view]}")
        env = self.session.env
        for column in range(env.cols):
            x = self.board.x + column * self.cell_size + self.cell_size // 2
            self.text(column, x, self.board.y - 11, 12, theme.MUTED, center=True)
        for row in range(env.rows):
            y = self.board.y + row * self.cell_size + self.cell_size // 2
            self.text(row, self.board.x - 12, y, 12, theme.MUTED, center=True)
        max_visits = max(1, int(self.session.agent.visits.sum(axis=1).max()))
        for state in range(env.observation_space.n):
            self.draw_tile(state, max_visits)

        trail = self.session.trail[-80:]
        current = self.session.state
        if self.watch is not None:
            trail = self.watch["path"][: self.watch_index + 1]
            current = self.watch["path"][self.watch_index]
        if self.view == 0 and len(trail) > 1:
            pg.draw.lines(self.window, theme.TRAIL, False, [self.coords(s) for s in trail], 2)
        x, y = self.coords(current)
        marker = (x + self.cell_size // 2 - 8, y - self.cell_size // 2 + 8)
        pg.draw.circle(self.window, theme.BLUE, marker, 5)
        self.text(
            "Click a tile to inspect its action values.",
            self.world_panel.x + 12,
            self.world_panel.bottom - 22,
            color=theme.MUTED,
        )

    def draw_table(self):
        self.panel(self.world_panel, "Q-table / Scroll to inspect all states")
        fractions = (0, 0.10, 0.26, 0.39, 0.52, 0.65, 0.78, 0.90)
        columns = [self.table_body.x + round(f * self.table_body.width) for f in fractions]
        headers = ("State", "(r, c)", "Tile", "Up", "Right", "Down", "Left", "Visits")
        for x, label in zip(columns, headers):
            self.text(label, x, self.table_body.y - 25, 13, theme.MUTED)
        stop = min(self.scroll + self.visible_rows, len(self.session.agent.q))
        for index, state in enumerate(range(self.scroll, stop)):
            row = pg.Rect(
                self.table_body.x,
                self.table_body.y + index * theme.ROW_HEIGHT,
                self.table_body.width,
                theme.ROW_HEIGHT,
            )
            if state == self.selected:
                pg.draw.rect(self.window, theme.SELECTION, row)
            pg.draw.line(self.window, theme.BORDER, row.bottomleft, row.bottomright)
            tile = self.session.env.tile(state)
            labels = (state, str(self.session.env.position(state)), theme.TILE_NAMES[tile])
            for x, label in zip(columns, labels):
                self.text(label, x, row.y + 6, 13, mono=True)
            for action in range(4):
                value = self.session.agent.q[state, action]
                label = "--" if tile in "#~G" else f"{value:.1f}"
                self.text(
                    label, columns[action + 3], row.y + 6, 13, self.value_color(value), mono=True
                )
            visits = int(self.session.agent.visits[state].sum())
            self.text(visits, columns[-1], row.y + 6, 13, theme.MUTED, mono=True)
        self.text(
            f"Rows {self.scroll + 1}-{stop} of {len(self.session.agent.q)}",
            self.world_panel.x + 12,
            self.world_panel.bottom - 22,
            color=theme.MUTED,
        )

    def draw_details(self):
        """Keep the board visible while choosing one supporting explanation."""
        x = self.detail_panel.x
        for label, name in (
            ("Learning curve", "chart"),
            ("Selected tile", "tile"),
            ("Learning step", "update"),
        ):
            x = self.button(label, x, name, self.detail == name, y=150)
        if self.detail == "tile":
            self.draw_inspector()
        elif self.detail == "update":
            self.draw_update()
        else:
            self.draw_chart()

    def draw_inspector(self):
        panel = self.inspector_panel
        state = self.selected
        agent = self.session.agent
        tile = self.session.env.tile(state)
        self.panel(
            panel, f"State {state} / {self.session.env.position(state)} / {theme.TILE_NAMES[tile]}"
        )
        x, y = panel.x + 12, panel.y + 46
        self.text("Action", x, y, color=theme.MUTED)
        self.text("Q-value", panel.right - 164, y, color=theme.MUTED)
        self.text("Visits", panel.right - 60, y, color=theme.MUTED)
        values = agent.q[state]
        scale = max(1.0, float(np.abs(values).max()))
        axis_x = panel.x + 145
        for action, name in enumerate(ACTION_NAMES):
            row_y = y + 27 + action * 28
            best = agent.visits[state].sum() > 0 and values[action] == values.max()
            self.text(name.capitalize(), x, row_y, color=theme.BLUE if best else theme.TEXT)
            pg.draw.line(
                self.window, theme.BORDER, (axis_x - 40, row_y + 8), (axis_x + 40, row_y + 8), 4
            )
            pg.draw.line(self.window, theme.MUTED, (axis_x, row_y + 1), (axis_x, row_y + 15))
            value = float(values[action])
            if value:
                end_x = axis_x + round(value / scale * 40)
                pg.draw.line(
                    self.window, self.value_color(value), (axis_x, row_y + 8), (end_x, row_y + 8), 4
                )
            self.text(f"{value:7.2f}", panel.right - 174, row_y, mono=True)
            self.text(int(agent.visits[state, action]), panel.right - 60, row_y, mono=True)
        note = (
            "No decisions on blocked or terminal tiles."
            if tile in "#~G"
            else "Q = estimated discounted return"
        )
        self.text(note, x, y + 151, 13, theme.MUTED)

    def draw_update(self):
        panel = self.update_panel
        self.panel(panel, "Last Q-learning update")
        x, y = panel.x + 12, panel.y + 45
        update = self.session.last_update
        if update is None:
            self.text("Q(s, a) += alpha * (target - Q(s, a))", x, y, mono=True)
            self.text("target = reward + gamma * max Q(next state)", x, y + 27, 13, theme.MUTED)
            self.text("Press N to see one update.", x, y + 64, color=theme.MUTED)
            return
        config = self.session.config
        action = ACTION_NAMES[update["action"]].lower()
        self.text(
            f"s{update['state']} / {action} / s{update['next_state']} / reward {update['reward']:+.0f}",
            x,
            y,
        )
        self.text(
            f"target = {update['reward']:.0f} + {config.gamma:g} * {update['future']:.2f} = {update['target']:.2f}",
            x,
            y + 25,
            13,
            mono=True,
        )
        self.text(
            f"Q: {update['old']:.2f} + {config.alpha:g} * ({update['target']:.2f} - {update['old']:.2f}) = {update['new']:.2f}",
            x,
            y + 49,
            13,
            theme.BLUE,
            mono=True,
        )
        mode = "Explore" if update["exploratory"] else "Exploit"
        detail = update["event"]
        if update["terminated"]:
            detail = "goal / future return = 0"
        elif update["truncated"]:
            detail = "time limit / bootstrap kept"
        self.text(f"{mode} / {detail}", x, y + 78, 13, theme.MUTED)

    def draw_chart(self):
        panel = self.chart_panel
        self.panel(panel, "Episode return / Moving average (25)")
        history = self.session.history
        if not history:
            self.text(
                "Waiting for the first completed episode.",
                panel.x + 12,
                panel.y + 49,
                color=theme.MUTED,
            )
            return
        # Running totals also handle the first episodes, before the window is full.
        values = []
        total = 0.0
        for index, episode in enumerate(history):
            total += episode["return"]
            if index >= 25:
                total -= history[index - 25]["return"]
            values.append(total / min(index + 1, 25))
        low, high = min(min(values), 0), max(max(values), 1)
        plot = pg.Rect(panel.x + 59, panel.y + 46, panel.width - 76, panel.height - 73)
        for value in (low, high):
            y = plot.bottom - round((value - low) / (high - low) * plot.height)
            pg.draw.line(self.window, theme.BORDER, (plot.x, y), (plot.right, y))
            self.text(round(value), panel.x + 9, y - 7, 12, theme.MUTED)
        indices = np.linspace(0, len(values) - 1, min(len(values), plot.width), dtype=int)
        points = [
            (
                plot.x + round(i / max(1, len(values) - 1) * plot.width),
                plot.bottom - round((values[i] - low) / (high - low) * plot.height),
            )
            for i in indices
        ]
        if len(points) > 1:
            pg.draw.lines(self.window, theme.BLUE, False, points, 2)
        else:
            pg.draw.circle(self.window, theme.BLUE, points[0], 3)
        self.text("1", plot.x, plot.bottom + 7, 12, theme.MUTED)
        self.text(f"Episode {len(history):,}", plot.right - 105, plot.bottom + 7, 12, theme.MUTED)

    def draw_notes(self):
        panel = self.legend_panel
        x, y = panel.x, panel.y
        self.text("Rewards", x, y, bold=True)
        rewards = self.session.env.rewards
        self.text(
            f"Path {rewards['path']:+g}   Trap {rewards['trap']:+g}   Goal {rewards['goal']:+g}   Blocked {rewards['blocked']:+g}",
            x,
            y + 24,
        )
        self.text(
            "S start   G goal   T trap   # wall   ~ river   /   Blue dot: agent",
            x,
            y + 49,
            13,
            theme.MUTED,
        )
        panel = self.settings_panel
        x, y = panel.x, panel.y
        config = self.session.config
        self.text(f"alpha {config.alpha:g}    gamma {config.gamma:g}    seed {config.seed}", x, y)
        self.text(
            f"Epsilon decay {config.epsilon_decay:g} / minimum {config.epsilon_min:g}",
            x,
            y + 24,
            color=theme.MUTED,
        )
        self.text(
            f"Episode: {self.session.env.steps}/{self.session.env.max_steps} steps / return {self.session.episode_return:+.0f}",
            x,
            y + 49,
            13,
            theme.MUTED,
        )

    def draw(self):
        # A window can also resize through the OS or move to another display.
        size = tuple(
            max(minimum, actual)
            for minimum, actual in zip(theme.MIN_SIZE, self.display.native_window.size)
        )
        self.display.resize(size)
        self.window = self.display.surface
        self.update_layout()
        self.display.labels.clear()
        self.window.fill(theme.BACKGROUND)
        self.buttons.clear()
        self.draw_header()
        if self.view == 3:
            self.draw_table()
        else:
            self.draw_board()
        self.draw_details()
        self.draw_notes()
        width, height = self.window.get_size()
        pg.draw.line(self.window, theme.BORDER, (0, height - 29), (width, height - 29))
        self.text(self.status, theme.MARGIN, height - 21, 13, theme.MUTED, max_width=width - 120)
        self.text("Esc: quit", width - 70, height - 21, 13, theme.MUTED)
        self.display.present()
        if self.diagnose_display and time.monotonic() >= self.next_diagnostic:
            from .diagnostics import save

            try:
                path = save(self.display, self.output)
                print(f"Live display diagnostics: {path.resolve()}", flush=True)
            except (OSError, AttributeError, RuntimeError) as error:
                print(f"Display diagnostics failed: {error}", flush=True)
                self.diagnose_display = False
            self.next_diagnostic = time.monotonic() + 5

    def stop_training(self):
        self.running = False
        self.target = None
        self.credit = 0.0

    def action(self, action):
        """Toolbar and keyboard commands share the same code path."""
        if action in ("chart", "tile", "update"):
            self.detail = action
        elif action == "view":
            self.view = (self.view + 1) % len(theme.VIEWS)
        elif action == "speed":
            self.speed = (self.speed + 1) % len(theme.SPEEDS)
        elif action == "export":
            self.export()
        elif action == "reset":
            self.reset()
        elif action == "watch":
            self.start_playback()
        elif action == "run":
            was_running = self.running
            self.stop_training()
            self.watch = None
            self.running = not was_running
            self.status = "Training." if self.running else "Paused."
        elif action == "step":
            self.stop_training()
            self.watch = None
            update = self.session.step()
            self.selected = update["state"]
            self.detail = "update"
            self.status = "One transition learned. The updated state is selected."
        elif action == "batch":
            self.watch = None
            self.target = len(self.session.history) + 500
            self.running = True
            self.status = f"Training until episode {self.target:,}."

    def reset(self):
        env = self.session.env
        self.session = TrainingSession(AdventureEnv(env.layout, env.max_steps), self.session.config)
        self.stop_training()
        self.watch = None
        self.selected = self.session.env.start
        self.scroll = 0
        self.status = "Reset. Same map and seed; all Q-values are zero."

    def export(self):
        try:
            path = self.session.export(self.output)
            self.status = f"Exported Q-table, episode metrics and evaluation to {path.resolve()}"
        except OSError as error:
            self.status = f"Export failed: {error}"

    def start_playback(self):
        self.stop_training()
        self.watch = self.session.greedy_rollout()
        self.watch_index = 0
        self.watch_elapsed = 0.0
        result = self.watch
        self.status = (
            f"Greedy playback: {result['reason']} / {len(result['path']) - 1} moves / "
            f"return {result['return']:+.0f}. Q-table frozen."
        )

    def handle_click(self, position):
        for rect, action in self.buttons:
            if rect.collidepoint(position):
                self.action(action)
                return
        x, y = position
        if self.view == 3 and self.table_body.collidepoint(position):
            state = self.scroll + (y - self.table_body.y) // theme.ROW_HEIGHT
            if state < len(self.session.agent.q):
                self.selected = state
                self.detail = "tile"
        elif self.view != 3 and self.board.collidepoint(position):
            row = (y - self.board.y) // self.cell_size
            column = (x - self.board.x) // self.cell_size
            self.selected = row * self.session.env.cols + column
            self.detail = "tile"

    def handle_event(self, event):
        if event.type in (pg.QUIT, pg.WINDOWCLOSE):
            self.alive = False
        elif event.type == pg.VIDEORESIZE:
            size = (max(theme.MIN_SIZE[0], event.w), max(theme.MIN_SIZE[1], event.h))
            self.display.resize(size)
            self.window = self.display.surface
            self.update_layout()
        elif event.type == pg.KEYDOWN:
            shortcuts = {
                pg.K_SPACE: "run",
                pg.K_n: "step",
                pg.K_w: "watch",
                pg.K_q: "view",
                pg.K_e: "export",
                pg.K_r: "reset",
            }
            if event.key == pg.K_ESCAPE:
                self.alive = False
            elif event.key in shortcuts:
                self.action(shortcuts[event.key])
            elif pg.K_1 <= event.key <= pg.K_4:
                self.speed = event.key - pg.K_1
        elif event.type == pg.MOUSEWHEEL and self.view == 3:
            self.scroll -= event.y * 3
            self.clamp_scroll()
        elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            self.handle_click(event.pos)

    def advance(self, dt):
        """Leave time for UI events even during accelerated training."""
        if self.watch is not None:
            self.watch_elapsed += dt
            if self.watch_elapsed >= 0.22:
                self.watch_elapsed = 0.0
                self.watch_index = min(self.watch_index + 1, len(self.watch["path"]) - 1)
        if not self.running:
            return
        self.credit = min(3000, self.credit + dt * theme.SPEEDS[self.speed])
        budget = 3000 if self.target is not None else int(self.credit)
        deadline = time.perf_counter() + 0.012
        for _ in range(budget):
            self.session.step()
            if self.target is None:
                self.credit -= 1
            if self.target is not None and len(self.session.history) >= self.target:
                self.stop_training()
                result = self.session.greedy_rollout()
                self.status = (
                    f"Training complete: {result['reason']} / "
                    f"{len(result['path']) - 1} moves / return {result['return']:+.0f}. Press W to watch."
                )
                break
            if time.perf_counter() >= deadline:
                break

    def run(self):
        clock = pg.time.Clock()
        try:
            while self.alive:
                dt = min(clock.tick(60) / 1000, 0.1)
                for event in pg.event.get():
                    self.handle_event(event)
                self.advance(dt)
                self.draw()
        finally:
            self.display.close()
            pg.quit()

    def screenshot(self, path):
        try:
            self.draw()
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            pg.image.save(self.display.frame, str(path))
        finally:
            self.display.close()
            pg.quit()
