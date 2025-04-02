import numpy as np
import pytest

from pygridagg.aggregate import FlexibleGridLayout, SquareGridLayout, PointAggregator
from pygridagg.examples import load_japanese_earthquake_data

np.random.seed(42)


def test_empty_point_data():
    empty_coords = np.array([]).reshape(0, 2)
    layout = SquareGridLayout(10, grid_center=(0, 0), num_cells=5 ** 2)
    agg = PointAggregator(layout, empty_coords)

    assert np.all(agg.cell_aggregates == 0)
    assert agg.grid_col_ids.size == 0
    assert agg.grid_row_ids.size == 0
    assert agg.inside_mask.size == 0


def test_localisation_only():
    N = 10
    rand_coords = np.random.randn(N, 2)
    layout = SquareGridLayout(1, grid_center=(0.5, 0.5), num_cells=5 ** 2)
    agg = PointAggregator(layout, rand_coords, localise_only=True)

    assert agg.cell_aggregates is None
    assert agg.grid_col_ids.shape == (N,)
    assert agg.grid_row_ids.shape == (N,)
    assert agg.inside_mask.shape == (N,)


def test_aggregator_shapes():
    N = 25
    C, R = 7, 13

    rand_coords = np.random.randn(N, 2)
    layout = FlexibleGridLayout(10, 10, grid_center=(0, 0), num_cols=C, num_rows=R)

    agg = PointAggregator(layout, rand_coords)
    assert agg.cell_aggregates.shape == layout.shape == (R, C)
    assert agg.grid_col_ids.shape == (N,)
    assert agg.grid_row_ids.shape == (N,)
    assert agg.inside_mask.shape == (N,)


def test_no_points_lost_no_oob_quakes():
    coords, _ = load_japanese_earthquake_data()
    N = coords.shape[0]

    layout = SquareGridLayout.from_points(
        coords,
        padding_percentage=0.001,
        num_cells=20 ** 2
    )

    agg = PointAggregator(layout, coords)
    assert agg.cell_aggregates.sum() == N


def test_no_points_lost_with_oob_quakes():
    coords, _ = load_japanese_earthquake_data()
    N = coords.shape[0]

    # Below, we construct a square grid centred on Tokyo
    # while the full earthquake data extends much further
    # into the Japanese archipelago.
    # Note: Coordinates are in degrees lon/lat.

    tokyo = 139.753481, 35.684568
    layout = SquareGridLayout(
        10,
        grid_center=tokyo,
        num_cells=50 ** 2
    )

    # check the number of quakes outside the grid bounds
    x = coords[:, 0]
    y = coords[:, 1]
    ood_x = (x < layout.x_min) | (x > layout.x_max)
    ood_y = (y < layout.y_min) | (y > layout.y_max)
    num_out_of_domain = (ood_x | ood_y).sum()  # type: ignore
    num_inside_domain = N - num_out_of_domain
    assert num_out_of_domain > 0  # ensure we actually have out-of-bounds quakes

    agg = PointAggregator(layout, coords, warn_out_of_bounds=False)
    assert agg.cell_aggregates.sum() == num_inside_domain


def test_bespoke_example1_results_no_weights(
        bespoke_test_points1,
        bespoke_test_layout1,
):
    expected_counts = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 2],
        [1, 0, 0, 0]]
    )

    agg = PointAggregator(bespoke_test_layout1, bespoke_test_points1)
    assert np.all(agg.cell_aggregates == expected_counts)


def test_bespoke_example2_results_no_weights(
        bespoke_test_points2,
        bespoke_test_layout2,
):
    expected_counts = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
    ]
    )

    layout = SquareGridLayout(2, grid_center=(-1, -1), num_cells=36)
    agg = PointAggregator(bespoke_test_layout2, bespoke_test_points2)
    assert np.all(agg.cell_aggregates == expected_counts)


def test_bespoke_example1_results_with_weights(
        bespoke_test_points1,
        bespoke_test_layout1
):
    point_weights = np.array([5, 4, 3, 2, 1])

    expected_weighted_sums = np.array([
        [0, 4, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 3],
        [5, 0, 0, 0]],
        dtype='float'
    )

    agg = PointAggregator(
        bespoke_test_layout1,
        bespoke_test_points1,
        point_weights=point_weights
    )
    assert np.all(agg.cell_aggregates == expected_weighted_sums)


def test_bespoke_example2_results_with_weights(
        bespoke_test_points2,
        bespoke_test_layout2
):
    point_weights = np.array([-1, +2, -3, +4, -5])

    expected_weighted_sums = np.array([
        [-1, 0, 0, 0,  0, 0],
        [ 0, 0, 0, 0, -1, 0],
        [ 0, 0, 0, 0,  0, 0],
        [ 0, 0, 0, 0,  0, 0],
        [ 0, 0, 2, 0,  0, 0],
        [ 0, 0, 0, 0, -3, 0]],
        dtype = 'float'
    )

    agg = PointAggregator(
        bespoke_test_layout2,
        bespoke_test_points2,
        point_weights=point_weights
    )
    assert np.all(agg.cell_aggregates == expected_weighted_sums)


@pytest.fixture
def bespoke_test_layout1():
    return SquareGridLayout(1, grid_center=(0.5, 0.5), num_cells=16)


@pytest.fixture
def bespoke_test_layout2():
    return SquareGridLayout(2, grid_center=(-1, -1), num_cells=36)


@pytest.fixture
def bespoke_test_points1():
    # constructed assuming a 4x4 square grid on unit square
    return np.array([
        [0.10, 1.00],
        [0.30, 0.10],
        [0.60, 0.40],
        [0.75, 0.51],
        [1.00, 0.74],
    ])


@pytest.fixture
def bespoke_test_points2():
    # constructed assuming 6x6 square grid with bounding box [-2, -2, 0, 0]
    return np.array([
        [-2.00, -2.00],
        [-1.30, -0.50],
        [-0.50, -0.01],
        [-0.66, -1.34],
        [-0.50, -1.50],
    ])
