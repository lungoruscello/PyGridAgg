import numpy as np
import pytest

from pygridagg.aggregate import FlexibleGridLayout, SquareGridLayout


def test_non_square_grid_construction():
    W, H = 10, 7
    C, R = 5, 10
    layout = FlexibleGridLayout(
        W, H, grid_center=(0, 0), num_cols=C, num_rows=R
    )

    assert layout.cell_width == W / C
    assert layout.cell_height == H / R
    assert layout.shape == (R, C)


def test_square_grid_construction():
    L = 10
    C = 5
    layout = SquareGridLayout(
        L, grid_center=(0, 0), num_cells=C ** 2
    )
    assert layout.shape == (C, C)
    assert layout.cell_width == L / C
    assert layout.cell_height == L / C


def test_bounds_inference():
    point_coords = np.array([[0, 4], [7, 9]])
    layout = SquareGridLayout.from_points(
        point_coords,
        num_cells=9,
        padding_percentage=0
    )

    assert layout.x_min == 0
    assert layout.x_max == 7
    assert layout.y_min == 3
    assert layout.y_max == 10

    empty_coords = np.array([]).reshape(0, 2)
    with pytest.raises(ValueError):
        SquareGridLayout.from_points(empty_coords, num_cells=9)
