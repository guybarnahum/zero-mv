
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List, Tuple

def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")

def save_image(img: Image.Image, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)

def annotate(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.rectangle([0, 0, img.width, 22], fill=(0,0,0))
    draw.text((5, 3), text, fill=(255,255,255), font=font)
    return img

def make_contact_sheet(images: list[Image.Image], cols: int) -> Image.Image:
    if not images:
        raise ValueError("No images to grid.")
    w, h = images[0].size
    rows = int(np.ceil(len(images) / cols))
    sheet = Image.new("RGB", (cols * w, rows * h), (0,0,0))
    for i, im in enumerate(images):
        r = i // cols
        c = i % cols
        sheet.paste(im, (c * w, r * h))
    return sheet

def to_square(img: Image.Image, min_side: int = 320) -> Image.Image:
    """Pad to square (using edge pixel as bg); ensure at least min_side."""
    w, h = img.size
    side = max(w, h, min_side)
    bg = Image.new("RGB", (side, side), img.getpixel((0, 0)) if w and h else (0, 0, 0))
    bg.paste(img, ((side - w) // 2, (side - h) // 2))
    return bg

def try_split_grid(grid: Image.Image) -> List[Image.Image]:
    """Split a Zero123++ multi-view grid into 6 tiles if layout matches.
    Tries (6x1, 3x2, 2x3, 1x6). Falls back to the original grid as a single tile.
    """
    W, H = grid.size
    candidates: List[Tuple[int, int]] = [(6, 1), (3, 2), (2, 3), (1, 6)]
    for cols, rows in candidates:
        if W % cols or H % rows:
            continue
        tile_w, tile_h = W // cols, H // rows
        tiles: List[Image.Image] = []
        for r in range(rows):
            for c in range(cols):
                left, upper = c * tile_w, r * tile_h
                tiles.append(grid.crop((left, upper, left + tile_w, upper + tile_h)))
        if len(tiles) == 6 and all(t.size == (tile_w, tile_h) for t in tiles):
            return tiles
    return [grid]
