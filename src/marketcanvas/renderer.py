from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from marketcanvas.canvas import Canvas
from marketcanvas.contrast import hex_to_rgb
from marketcanvas.elements import ElementType


def render_to_image(canvas: Canvas) -> Image.Image:
    """Render canvas to a PIL Image."""
    img = Image.new("RGB", (canvas.width, canvas.height), hex_to_rgb(canvas.background))
    draw = ImageDraw.Draw(img)

    for el in canvas.elements:
        color = hex_to_rgb(el.color)

        if el.type == ElementType.SHAPE:
            draw.rectangle([el.x, el.y, el.right - 1, el.bottom - 1], fill=color)
            if el.content:
                _draw_centered_text(draw, el, hex_to_rgb(el.text_color))

        elif el.type == ElementType.TEXT:
            draw.rectangle([el.x, el.y, el.right - 1, el.bottom - 1], fill=color)
            _draw_centered_text(draw, el, hex_to_rgb(el.text_color))

        elif el.type == ElementType.IMAGE:
            draw.rectangle([el.x, el.y, el.right - 1, el.bottom - 1], fill=color)
            draw.rectangle([el.x, el.y, el.right - 1, el.bottom - 1], outline=(128, 128, 128))
            if el.content:
                _draw_centered_text(draw, el, (200, 200, 200))

    return img


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    el: "Element",  # noqa: F821
    color: tuple[int, int, int],
) -> None:
    try:
        size = max(12, min(el.height // 2, 48))
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), el.content, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = el.x + (el.width - tw) / 2
    ty = el.y + (el.height - th) / 2
    draw.text((tx, ty), el.content, fill=color, font=font)


def render_to_array(canvas: Canvas) -> np.ndarray:
    """Render canvas to RGB numpy array of shape (H, W, 3)."""
    return np.array(render_to_image(canvas))


def render_to_base64(canvas: Canvas) -> str:
    """Render canvas to base64-encoded PNG string."""
    img = render_to_image(canvas)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def save_png(canvas: Canvas, path: str) -> None:
    """Save canvas render to a PNG file."""
    render_to_image(canvas).save(path)
