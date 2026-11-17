"""Unit tests for collage generator."""

from __future__ import annotations

import numpy as np
import pytest

from digital_attendance.data.collage import compose_grid_collage


def test_compose_grid_collage() -> None:
    """Verify composition of a 2x2 grid collage."""
    tile = np.ones((64, 64), dtype=np.uint8) * 100
    tiles = [tile * (i + 1) for i in range(4)]
    labels = ["P1", "P2", "P3", "P4"]

    res = compose_grid_collage(tiles, labels, grid_rows=2, grid_cols=2, tile_size=(64, 64))

    assert res.image.shape == (128, 128)
    assert len(res.cells) == 4

    cell0 = res.cells[0]
    assert cell0.label == "P1"
    assert (cell0.x, cell0.y, cell0.w, cell0.h) == (0, 0, 64, 64)

    cell3 = res.cells[3]
    assert cell3.label == "P4"
    assert (cell3.x, cell3.y, cell3.w, cell3.h) == (64, 64, 64, 64)


def test_compose_grid_collage_insufficient_tiles() -> None:
    """Verify error raised when fewer tiles than required are provided."""
    tiles = [np.zeros((32, 32))]
    labels = ["P1"]
    with pytest.raises(ValueError):
        compose_grid_collage(tiles, labels, grid_rows=2, grid_cols=2)
