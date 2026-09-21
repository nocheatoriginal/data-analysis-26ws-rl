"""HiDPI window support, kept separate from the dashboard and learning code.

Window coordinates are logical points; a Retina window can have twice as many
physical pixels in each direction. Shapes use the existing logical surface.
Text is drawn afterwards at the actual output resolution, never enlarged from
a low-resolution glyph bitmap.
"""

from dataclasses import dataclass

import pygame as pg
import pygame.ftfont as ftfont
from pygame._sdl2.video import Renderer, Texture, Window

from . import theme


@dataclass
class TextLabel:
    value: str
    position: tuple
    size: int
    color: str
    bold: bool
    mono: bool
    center: bool


class Display:
    def __init__(self, size):
        ftfont.init()
        self.native_window = Window(
            "Q-learning adventure", size=size, resizable=True, allow_highdpi=True
        )
        self.renderer = Renderer(self.native_window)
        self.surface = pg.Surface(size)
        self.labels = []
        self.fonts = {}
        self.frame = None
        self.texture = None

    def font(self, size=14, bold=False, mono=False):
        key = (size, bold, mono)
        if key not in self.fonts:
            family = "Menlo,Consolas,DejaVu Sans Mono" if mono else "Arial,Segoe UI,DejaVu Sans"
            font = ftfont.SysFont(family, size, bold=bold)
            # A small weight increase keeps thin strokes legible on dark tiles.
            # Use the same metrics for layout and rendering, including at 2x DPI.
            if not bold:
                font.strong = True
                font.strength = theme.TEXT_STRENGTH
            self.fonts[key] = font
        return self.fonts[key]

    def resize(self, size):
        if self.native_window.size != size:
            self.native_window.size = size
        if self.surface.get_size() != size:
            self.surface = pg.Surface(size)

    def compose(self, pixel_size=None):
        """Build a frame in physical pixels. An explicit size is useful for tests."""
        if pixel_size is None:
            # With no logical_size or custom viewport, SDL reports output pixels.
            pixel_size = self.renderer.get_viewport().size
        if self.frame is None or self.frame.get_size() != pixel_size:
            self.frame = pg.Surface(pixel_size)
        pg.transform.scale(self.surface, pixel_size, self.frame)
        scale_x = pixel_size[0] / self.surface.get_width()
        scale_y = pixel_size[1] / self.surface.get_height()

        for label in self.labels:
            font = self.font(max(1, round(label.size * scale_y)), label.bold, label.mono)
            glyphs = font.render(label.value, True, label.color)
            position = (round(label.position[0] * scale_x), round(label.position[1] * scale_y))
            rect = glyphs.get_rect()
            if label.center:
                rect.center = position
            else:
                rect.topleft = position
            self.frame.blit(glyphs, rect)
        return self.frame

    def present(self):
        frame = self.compose()
        if self.texture is None or self.texture.get_rect().size != frame.get_size():
            self.texture = Texture(self.renderer, frame.get_size(), streaming=True)
        self.texture.update(frame)
        self.renderer.clear()
        self.texture.draw()
        self.renderer.present()

    def close(self):
        # Release SDL resources before shutting down Pygame.
        self.texture = None
        self.renderer = None
        self.native_window.destroy()
