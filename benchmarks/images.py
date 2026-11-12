"""Deterministic Pillow renderings of the image-case specs.

Images are generated from the spec in `cases_image.py` at run time rather
than stored as binaries, so the dataset stays reviewable as text and every
model receives byte-identical pixels.
"""

from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image, ImageDraw, ImageFont

_COLORS = {
    "red": (220, 40, 40), "blue": (40, 80, 220), "green": (40, 160, 70),
    "yellow": (240, 200, 30), "black": (20, 20, 20), "orange": (245, 140, 30),
    "purple": (140, 60, 180),
}


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_card(lines: list[str]) -> Image.Image:
    font = _font(28)
    width, height = 900, 80 + 44 * len(lines)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([4, 4, width - 5, height - 5], outline=(60, 60, 60), width=3)
    for i, line in enumerate(lines):
        draw.text((36, 40 + 44 * i), line, fill="black", font=font)
    return img


def _table(header: list[str], rows: list[list[str]]) -> Image.Image:
    font = _font(26)
    col_w, row_h = 220, 46
    cols = len(header)
    width, height = 40 + col_w * cols, 40 + row_h * (len(rows) + 1)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    for r, row in enumerate([header, *rows]):
        y = 20 + r * row_h
        if r == 0:
            draw.rectangle([20, y, 20 + col_w * cols, y + row_h], fill=(225, 225, 240))
        for c in range(cols):
            x = 20 + c * col_w
            draw.rectangle([x, y, x + col_w, y + row_h], outline="black", width=2)
            text = row[c] if c < len(row) else ""
            draw.text((x + 12, y + 9), text, fill="black", font=font)
    return img


def _shapes(shapes: list[dict[str, str]]) -> Image.Image:
    size, gap = 110, 30
    width = gap + (size + gap) * len(shapes)
    img = Image.new("RGB", (max(width, 200), size + 2 * gap), "white")
    draw = ImageDraw.Draw(img)
    for i, shape in enumerate(shapes):  # left to right, in list order
        x0, y0 = gap + i * (size + gap), gap
        box = [x0, y0, x0 + size, y0 + size]
        fill = _COLORS.get(shape.get("color", "black"), (20, 20, 20))
        kind = shape.get("type")
        if kind == "circle":
            draw.ellipse(box, fill=fill)
        elif kind == "triangle":
            draw.polygon([(x0 + size / 2, y0), (x0, y0 + size), (x0 + size, y0 + size)], fill=fill)
        else:
            draw.rectangle(box, fill=fill)
    return img


def render(spec: dict[str, Any]) -> Image.Image:
    kind = spec.get("kind")
    if kind == "text_card":
        return _text_card(spec["lines"])
    if kind == "table":
        return _table(spec["header"], spec["rows"])
    if kind == "shapes":
        return _shapes(spec["shapes"])
    raise ValueError(f"unknown image kind {kind!r}")


def to_data_url(spec: dict[str, Any]) -> str:
    buffer = io.BytesIO()
    render(spec).save(buffer, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


__all__ = ["render", "to_data_url"]
