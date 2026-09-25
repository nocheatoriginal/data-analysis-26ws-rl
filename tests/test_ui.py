"""Exercise real Pygame rendering and controls using an offscreen display."""

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import unittest

import pygame as pg

from src.model.learning import TrainingSession
from src.ui import theme
from src.ui.dashboard import Dashboard


class DashboardTests(unittest.TestCase):
    def test_views_controls_selection_and_playback(self):
        dashboard = Dashboard(TrainingSession())
        try:
            dashboard.draw()
            dashboard.handle_event(pg.event.Event(pg.KEYDOWN, key=pg.K_n))
            self.assertEqual(dashboard.session.total_steps, 1)
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
            dashboard.action("watch")
            before = dashboard.session.total_steps
            dashboard.advance(1)
            dashboard.draw()
            self.assertEqual(dashboard.session.total_steps, before)
            dashboard.action("run")
            dashboard.advance(0.1)
            self.assertGreater(dashboard.session.total_steps, before)
            dashboard.action("reset")
            self.assertFalse(dashboard.running)
            self.assertEqual(dashboard.session.total_steps, 0)
            self.assertIsNone(dashboard.watch)
            dashboard.action("batch")
            # Exercise automatic stopping at an episode boundary cheaply.
            dashboard.target = 1
            for _ in range(100):
                dashboard.advance(0.1)
                if not dashboard.running:
                    break
            self.assertFalse(dashboard.running)
            self.assertEqual(len(dashboard.session.history), 1)
        finally:
            dashboard.display.close()
            pg.quit()

    def test_resize_keeps_text_size_and_click_targets(self):
        dashboard = Dashboard(TrainingSession())
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
