"""Exercise real Pygame rendering and controls using an offscreen display."""

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import tempfile
import unittest

import numpy as np
import pygame as pg

from src.model.checkpoints import train_checkpoints
from src.ui import theme
from src.ui.dashboard import Dashboard


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = tempfile.TemporaryDirectory()
        train_checkpoints(cls.output.name, checkpoints=(1, 50, 100))

    @classmethod
    def tearDownClass(cls):
        cls.output.cleanup()

    def test_views_controls_selection_and_playback(self):
        dashboard = Dashboard(self.output.name)
        try:
            before = dashboard.session.total_steps
            q = dashboard.session.agent.q.copy()
            dashboard.draw()
            dashboard.handle_event(pg.event.Event(pg.KEYDOWN, key=pg.K_SPACE))
            dashboard.advance(1 / theme.SPEEDS[dashboard.speed])
            dashboard.handle_event(pg.event.Event(pg.KEYDOWN, key=pg.K_SPACE))
            self.assertEqual(dashboard.session.total_steps, before)
            self.assertEqual(dashboard.watch_index, 1)
            self.assertEqual(dashboard.detail, "update")
            for view in range(4):
                dashboard.view = view
                dashboard.draw()
            dashboard.handle_event(pg.event.Event(pg.MOUSEWHEEL, y=-100))
            self.assertEqual(dashboard.scroll, 80 - dashboard.visible_rows)
            dashboard.draw()
            click = (dashboard.table_body.x + 10, dashboard.table_body.y + 10)
            dashboard.handle_event(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=click))
            self.assertEqual(dashboard.selected, dashboard.scroll)
            dashboard.action("play")
            dashboard.advance(1)
            dashboard.draw()
            self.assertEqual(dashboard.session.total_steps, before)
            np.testing.assert_array_equal(dashboard.session.agent.q, q)
            dashboard.action("restart")
            self.assertFalse(dashboard.running)
            self.assertEqual(dashboard.watch_index, 0)
            dashboard.action("mode")
            self.assertEqual(dashboard.playback_mode, "route")
            dashboard.action("play")
            dashboard.advance(1000)
            self.assertFalse(dashboard.running)
            self.assertEqual(dashboard.watch_index, len(dashboard.watch["steps"]))
            np.testing.assert_array_equal(dashboard.session.agent.q, q)
            dashboard.action("checkpoint:50")
            self.assertEqual(len(dashboard.session.history), 50)
            self.assertEqual(dashboard.watch_index, 0)
            self.assertFalse(dashboard.session.agent.q.flags.writeable)
            dashboard.handle_event(pg.event.Event(pg.KEYDOWN, key=pg.K_3))
            self.assertEqual(dashboard.checkpoint, 100)
            self.assertEqual(len(dashboard.session.history), 100)
        finally:
            dashboard.display.close()
            pg.quit()

    def test_playback_speeds_and_pause(self):
        dashboard = Dashboard(self.output.name)
        try:
            for index, moves in enumerate((1, 5, 10, 20)):
                dashboard.speed = index
                dashboard.action("restart")
                dashboard.action("play")
                dashboard.advance(1.001)
                self.assertEqual(dashboard.watch_index, moves)
                dashboard.action("play")
                dashboard.advance(1)
                self.assertEqual(dashboard.watch_index, moves)
            dashboard.draw()
            self.assertNotIn("step", [action for _, action in dashboard.buttons])
        finally:
            dashboard.display.close()
            pg.quit()

    def test_adventure_preserves_playback_and_shows_path_above_sprites(self):
        dashboard = Dashboard(self.output.name, checkpoint=100)
        try:
            dashboard.action("mode")
            dashboard.action("play")
            dashboard.advance(1)
            index = dashboard.watch_index
            q = dashboard.session.agent.q.copy()
            dashboard.view = 3
            dashboard.draw()
            button = next(rect for rect, action in dashboard.buttons if action == "adventure")
            dashboard.handle_click(button.center)
            self.assertTrue(dashboard.adventure)
            self.assertTrue(dashboard.running)
            self.assertEqual(dashboard.watch_index, index)
            for width, height in (theme.MIN_SIZE, theme.DEFAULT_SIZE, (1600, 1000)):
                dashboard.handle_event(pg.event.Event(pg.VIDEORESIZE, w=width, h=height))
                dashboard.draw()
                for rect, _ in dashboard.buttons:
                    self.assertTrue(dashboard.window.get_rect().contains(rect))
                # The path must be drawn on top of terrain, even when entered from Table.
                start, following = dashboard.watch["path"][:2]
                a, b = dashboard.coords(start), dashboard.coords(following)
                midpoint = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
                self.assertEqual(dashboard.window.get_at(midpoint), pg.Color("#ffe6a3"))
                dashboard.handle_click(dashboard.coords(23))
                self.assertEqual(dashboard.selected, 23)
                self.assertEqual(dashboard.detail, "tile")
            dashboard.handle_event(pg.event.Event(pg.KEYDOWN, key=pg.K_a))
            self.assertFalse(dashboard.adventure)
            self.assertEqual(dashboard.view, 3)
            self.assertEqual(dashboard.watch_index, index)
            np.testing.assert_array_equal(dashboard.session.agent.q, q)
        finally:
            dashboard.display.close()
            pg.quit()

    def test_resize_keeps_text_size_and_click_targets(self):
        dashboard = Dashboard(self.output.name)
        try:
            original_font_size = dashboard.font().size("Q-learning")
            for width, height in (theme.MIN_SIZE, theme.DEFAULT_SIZE, (1600, 1000)):
                dashboard.handle_event(pg.event.Event(pg.VIDEORESIZE, w=width, h=height))
                for view in range(4):
                    dashboard.view = view
                    dashboard.draw()
                    self.assertEqual(dashboard.window.get_size(), (width, height))
                    self.assertEqual(dashboard.font().size("Q-learning"), original_font_size)
                dashboard.view = 0
                dashboard.draw()
                dashboard.handle_click(dashboard.coords(23))
                self.assertEqual(dashboard.selected, 23)
                self.assertEqual(dashboard.detail, "tile")
                for detail in ("chart", "tile", "update"):
                    dashboard.action(detail)
                    dashboard.draw()
                    self.assertEqual(dashboard.detail, detail)
                    for rect, _ in dashboard.buttons:
                        self.assertTrue(dashboard.window.get_rect().contains(rect))
                dashboard.view = 3
                dashboard.draw()
                dashboard.handle_click(
                    (dashboard.table_body.x + 8, dashboard.table_body.y + theme.ROW_HEIGHT + 8)
                )
                self.assertEqual(dashboard.selected, dashboard.scroll + 1)
                self.assertTrue(dashboard.window.get_rect().contains(dashboard.buttons[-1][0]))
        finally:
            dashboard.display.close()
            pg.quit()


if __name__ == "__main__":
    unittest.main()
