import numpy as np
import pytest

from pygridagg.aggregate import FlexibleGridLayout, SquareGridLayout


def test_flexi_grid_bbox_construction():
    layout = FlexibleGridLayout(0, 1, 0, 2)

    assert layout.x_min == 0
    assert layout.x_max == 1
    assert layout.y_min == 0
    assert layout.y_max == 2


def test_flexi_grid_centroid_construction():
    W, H = 10, 7
    C, R = 5, 10
    layout = FlexibleGridLayout.from_centroid(
        W, H, grid_center=(0, 0), num_cols=C, num_rows=R
    )

    assert layout.cell_width == W / C
    assert layout.cell_height == H / R
    assert layout.shape == (R, C)


def test_flexi_grid_inferred_construction_no_pad(example_points):
    layout = FlexibleGridLayout.from_points(
        example_points, padding_percent=0
    )

    assert layout.x_min == 0
    assert layout.x_max == 7
    assert layout.y_min == 4
    assert layout.y_max == 9


def test_flexi_grid_inferred_construction_with_pad(example_points):
    layout = FlexibleGridLayout.from_points(
        example_points, padding_percent=10
    )

    assert np.isclose(layout.x_min, -0.35)
    assert np.isclose(layout.x_max, 7.35)
    assert np.isclose(layout.y_min, 3.75)
    assert np.isclose(layout.y_max, 9.25)


def test_square_grid_bbox_construction():
    L = 10
    C = 5
    layout = SquareGridLayout(0, 10, -10, -2, num_cells=C ** 2)
    assert layout.x_min == 0
    assert layout.x_max == 10
    assert layout.y_min == -11
    assert layout.y_max == -1
    assert layout.shape == (C, C)
    assert layout.cell_width == L / C
    assert layout.cell_height == L / C


def test_square_grid_centroid_construction():
    L = 10
    C = 5
    layout = SquareGridLayout.from_centroid(
        L, grid_center=(0, 0),  num_cells=C ** 2)
    assert layout.total_width == layout.total_width
    assert layout.x_min == -5
    assert layout.x_max == 5
    assert layout.y_min == -5
    assert layout.y_max == 5
    assert layout.shape == (C, C)
    assert layout.cell_width == L / C
    assert layout.cell_height == L / C


def test_square_grid_inferred_construction_no_pad(example_points):
    layout = SquareGridLayout.from_points(
        example_points, padding_percent=0
    )

    assert layout.x_min == 0
    assert layout.x_max == 7
    assert layout.y_min == 3
    assert layout.y_max == 10


def test_square_grid_inferred_construction_with_pad(example_points):
    layout = SquareGridLayout.from_points(
        example_points, padding_percent=10
    )

    assert np.isclose(layout.x_min, -0.35)
    assert np.isclose(layout.x_max, 7.35)
    assert np.isclose(layout.y_min, 2.65)
    assert np.isclose(layout.y_max, 10.35)


def test_nonsquare_cell_number_raises():
    with pytest.raises(ValueError):
        SquareGridLayout.from_centroid(1, grid_center=(0, 0), num_cells=3)


@pytest.fixture
def example_points():
    return np.array([[0, 4], [7, 9]])
