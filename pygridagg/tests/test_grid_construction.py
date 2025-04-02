import numpy as np
import pytest

from pygridagg.aggregate import GridLayout, SquareGridLayout, infer_spatial_bounds


def test_non_square_grid_construction():
    W, H = 10, 7
    C, R = 5, 10
    non_square_grid = GridLayout(W, H, grid_center=(0, 0), num_cols=C, num_rows=R)

    assert non_square_grid.cell_width == W / C
    assert non_square_grid.cell_height == H / R
    assert non_square_grid.shape == (R, C)


def test_square_grid_construction():
    L = 10
    C = 5
    grid = SquareGridLayout(L, grid_center=(0, 0), num_cells=C ** 2)
    assert grid.shape == (C, C)
    assert grid.cell_width == L / C
    assert grid.cell_height == L / C


def test_bounds_inference():
    point_coords = np.array([[0, 4], [7, 9]])
    grid = SquareGridLayout.from_points(point_coords, num_cells=3 ** 2)

    assert grid.x_min == 0
    assert grid.x_max == 7
    assert grid.y_min == 3
    assert grid.y_max == 10


def test_bounds_inference_empty_raises():
    empty_coords = np.array([]).reshape(0, 2)
    with pytest.raises(ValueError):
        SquareGridLayout.from_points(empty_coords, num_cells=3 ** 2)
