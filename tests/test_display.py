"""Check Retina rendering without depending on a physical Retina screen."""

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import unittest

import numpy as np
import pygame as pg

from adventure.display import Display, TextLabel


class DisplayTests(unittest.TestCase):
    def test_retina_text_is_rasterized_at_double_resolution(self):
        pg.init()
        display = Display((320, 160))
        try:
            display.surface.fill("black")
            display.labels.append(
                TextLabel("Q-value 32.82", (10, 12), 14, "white", False, True, False)
            )
            normal = display.compose((320, 160)).copy()
            retina = display.compose((640, 320)).copy()

            # The 2x frame must contain freshly rendered 28px glyphs exactly.
            expected = pg.Surface((640, 320))
            expected.fill("black")
            expected.blit(
                display.font(28, mono=True).render("Q-value 32.82", True, "white"), (20, 24)
            )
            np.testing.assert_array_equal(
                pg.surfarray.array3d(retina), pg.surfarray.array3d(expected)
            )
            enlarged = pg.transform.scale(normal, (640, 320))
            self.assertFalse(
                np.array_equal(pg.surfarray.array3d(retina), pg.surfarray.array3d(enlarged))
            )

            # Presentation follows the renderer's actual output size again.
            display.present()
            self.assertEqual(display.frame.get_size(), display.renderer.get_viewport().size)
        finally:
            display.close()
            pg.quit()
