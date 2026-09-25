"""Read the live SDL output configuration, without changing display settings.

This is an opt-in diagnostic for the real desktop window, not the dummy renderer.
Pygame 2.6 does not expose SDL_GetRendererOutputSize in its Python API, so ctypes
is used here only to query the SDL library already loaded by Pygame.
"""

import ctypes
import json
import os
import sys
from pathlib import Path

import pygame as pg


class RendererInfo(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("flags", ctypes.c_uint32),
        ("num_texture_formats", ctypes.c_uint32),
        ("texture_formats", ctypes.c_uint32 * 16),
        ("max_texture_width", ctypes.c_int),
        ("max_texture_height", ctypes.c_int),
    ]


def collect(display):
    """Return independent measurements of window, framebuffer, and texture size."""
    sdl = ctypes.CDLL(pg.base.__file__)
    pointer = ctypes.c_void_p
    int_pointer = ctypes.POINTER(ctypes.c_int)
    signatures = {
        "SDL_GetWindowFromID": ([ctypes.c_uint32], pointer),
        "SDL_GetRenderer": ([pointer], pointer),
        "SDL_GetRendererOutputSize": ([pointer, int_pointer, int_pointer], ctypes.c_int),
        "SDL_GetWindowSizeInPixels": ([pointer, int_pointer, int_pointer], None),
        "SDL_GetWindowFlags": ([pointer], ctypes.c_uint32),
        "SDL_GetRendererInfo": ([pointer, ctypes.POINTER(RendererInfo)], ctypes.c_int),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(sdl, name)
        function.argtypes = arguments
        function.restype = result

    window = sdl.SDL_GetWindowFromID(display.native_window.id)
    renderer = sdl.SDL_GetRenderer(window)
    if not window or not renderer:
        raise RuntimeError("The live SDL window or renderer could not be found.")

    output_width, output_height = ctypes.c_int(), ctypes.c_int()
    if sdl.SDL_GetRendererOutputSize(
        renderer, ctypes.byref(output_width), ctypes.byref(output_height)
    ):
        raise RuntimeError("SDL_GetRendererOutputSize failed.")
    window_width, window_height = ctypes.c_int(), ctypes.c_int()
    sdl.SDL_GetWindowSizeInPixels(window, ctypes.byref(window_width), ctypes.byref(window_height))
    info = RendererInfo()
    if sdl.SDL_GetRendererInfo(renderer, ctypes.byref(info)):
        raise RuntimeError("SDL_GetRendererInfo failed.")

    output_size = (output_width.value, output_height.value)
    window_size = display.native_window.size
    frame_size = display.frame.get_size() if display.frame is not None else None
    return {
        "python": sys.executable,
        "pygame_version": pg.version.ver,
        "pygame_path": pg.__file__,
        "sdl_version": pg.get_sdl_version(),
        "video_driver": pg.display.get_driver(),
        "renderer": info.name.decode(),
        "hardware_accelerated": bool(info.flags & 2),
        "window_hidpi_flag": bool(sdl.SDL_GetWindowFlags(window) & 0x2000),
        "window_points": window_size,
        "window_pixels": (window_width.value, window_height.value),
        "renderer_output_pixels": output_size,
        "renderer_viewport": tuple(display.renderer.get_viewport()),
        "renderer_scale": display.renderer.scale,
        "renderer_logical_size": display.renderer.logical_size,
        "frame_pixels": frame_size,
        "pixel_density": (output_size[0] / window_size[0], output_size[1] / window_size[1]),
        "frame_matches_output": frame_size == output_size,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "SDL_VIDEODRIVER",
                "SDL_VIDEO_HIGHDPI_DISABLED",
                "SDL_RENDER_DRIVER",
                "SDL_RENDER_SCALE_QUALITY",
            )
        },
    }


def save(display, output):
    report = collect(display)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "display-diagnostics.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path
