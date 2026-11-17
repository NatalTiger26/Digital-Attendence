"""Collage generator for simulating classroom / group attendance scenes.

Composes individual face images into rectangular grid collages (e.g. 4x4)
and maintains ground-truth spatial assignments for pipeline verification.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from digital_attendance.config import (
    COLLAGES_DIR,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_SHAPE,
    DEFAULT_IMAGE_WIDTH,
)
from digital_attendance.data.loader import resolve_data_path


@dataclass(frozen=True)
class CollageGridCell:
    """Ground-truth metadata for a tile placed within a synthesized collage.

    Attributes:
        label: Identity / subject name of the person in this grid cell.
        row: Zero-indexed row in the tile grid.
        col: Zero-indexed column in the tile grid.
        x: X-coordinate (pixels) of the top-left corner.
        y: Y-coordinate (pixels) of the top-left corner.
        w: Width (pixels) of the cell.
        h: Height (pixels) of the cell.
    """

    label: str
    row: int
    col: int
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class CollageResult:
    """Generated collage image bundled with spatial ground-truth annotations.

    Attributes:
        image: Synthesized composite image (BGR or Grayscale).
        cells: List of grid cells detailing which identity is in which location.
    """

    image: np.ndarray
    cells: list[CollageGridCell]


def compose_grid_collage(
    tiles: Sequence[np.ndarray],
    labels: Sequence[str],
    grid_rows: int = 4,
    grid_cols: int = 4,
    tile_size: tuple[int, int] = DEFAULT_IMAGE_SHAPE,
) -> CollageResult:
    """Assemble a list of face image tiles into a uniform 2D grid collage.

    Args:
        tiles: Sequence of images (each array of shape `(H, W)` or `(H, W, C)`).
        labels: Identity labels corresponding to each tile in `tiles`.
        grid_rows: Number of rows in the composite grid.
        grid_cols: Number of columns in the composite grid.
        tile_size: (width, height) in pixels to which each tile is normalized.

    Returns:
        CollageResult containing the composite image and cell bounding boxes.

    Raises:
        ValueError: If fewer tiles than `grid_rows * grid_cols` are provided.
    """
    needed = grid_rows * grid_cols
    if len(tiles) < needed:
        raise ValueError(f"Required {needed} tiles for a {grid_rows}x{grid_cols} grid, got {len(tiles)}.")
    if len(labels) < needed:
        raise ValueError(f"Required {needed} labels, got {len(labels)}.")

    target_w, target_h = tile_size
    cells: list[CollageGridCell] = []

    # Check channels
    sample = tiles[0]
    channels = sample.shape[2] if sample.ndim == 3 else 1

    total_h = grid_rows * target_h
    total_w = grid_cols * target_w

    if channels == 1:
        canvas = np.zeros((total_h, total_w), dtype=np.uint8)
    else:
        canvas = np.zeros((total_h, total_w, channels), dtype=np.uint8)

    idx = 0
    for r in range(grid_rows):
        for c in range(grid_cols):
            tile = tiles[idx]
            label = labels[idx]

            # Ensure correct resolution
            if tile.shape[1] != target_w or tile.shape[0] != target_h:
                tile = cv2.resize(tile, (target_w, target_h))

            # Channel alignment if needed
            if channels > 1 and tile.ndim == 2:
                tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
            elif channels == 1 and tile.ndim == 3:
                tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

            y_start = r * target_h
            y_end = y_start + target_h
            x_start = c * target_w
            x_end = x_start + target_w

            canvas[y_start:y_end, x_start:x_end] = tile

            cells.append(
                CollageGridCell(
                    label=label,
                    row=r,
                    col=c,
                    x=x_start,
                    y=y_start,
                    w=target_w,
                    h=target_h,
                )
            )
            idx += 1

    return CollageResult(image=canvas, cells=cells)


def save_collage(collage: np.ndarray, output_path: str | Path) -> Path:
    """Save a collage image to disk safely.

    Args:
        collage: Image array to write.
        output_path: Target filesystem path.

    Returns:
        Absolute resolved Path where image was written.
    """
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(out), collage)
    if not success:
        raise IOError(f"OpenCV failed writing image to: {out}")
    return out
