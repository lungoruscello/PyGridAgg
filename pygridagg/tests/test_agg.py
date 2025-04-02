from pathlib import Path

import numpy as np

from pygridagg.aggregate import GridLayout, SquareGridLayout, PointAggregator

_data_dir = Path(__file__).parent.parent / 'examples' / 'data'

np.random.seed(42)


def test_empty_point_data():
    empty_coords = np.array([]).reshape(0, 2)
    grid = SquareGridLayout(10, grid_center=(0, 0), num_cells=5 ** 2)
    agg = PointAggregator(grid, empty_coords)

    assert np.all(agg.cell_aggregates == 0)
    assert agg.grid_row_ids.size == 0
    assert agg.inside_mask.size == 0


def test_localisation_only():
    N = 10
    rand_coords = np.random.randn(N, 2)
    grid = SquareGridLayout(1, grid_center=(0.5, 0.5), num_cells=5 ** 2)
    loc = PointAggregator(grid, rand_coords, localise_only=True)

    assert loc.cell_aggregates is None
    assert loc.cell_aggregates is None
    assert loc.grid_col_ids.shape == (N,)
    assert loc.grid_row_ids.shape == (N,)


def test_aggregation_result_shapes():
    N = 25
    C, R = 7, 13

    rand_coords = np.random.randn(N, 2)
    non_square_grid = GridLayout(10, 10, grid_center=(0, 0), num_cols=C, num_rows=R)

    agg = PointAggregator(non_square_grid, rand_coords)
    assert agg.cell_aggregates.shape == non_square_grid.shape == (R, C)
    assert agg.grid_row_ids.shape == (N,)
    assert agg.inside_mask.shape == (N,)


def test_no_lost_points_no_oob():
    N = 100_000
    L = 20

    rand_coords = np.random.randn(N, 2)
    grid = SquareGridLayout(L, grid_center=(0, 0), num_cells=20 ** 2)

    # ensure simulated points are entirely inside the grid bounds
    half_length = L / 2
    assert rand_coords.min() > -half_length  # noqa
    assert rand_coords.max() < half_length  # noqa

    agg = PointAggregator(grid, rand_coords)
    assert agg.cell_aggregates.sum() == N


def test_no_lost_points_with_oob():
    N = 100_000
    L = 2

    rand_coords = np.random.randn(N, 2)
    grid = SquareGridLayout(L, grid_center=(0, 0), num_cells=20 ** 2)

    # check number of simulated points outside the grid bounds
    half_l = L / 2
    ood_x = (rand_coords[:, 0] < -half_l) | (rand_coords[:, 0] > half_l)
    ood_y = (rand_coords[:, 1] < -half_l) | (rand_coords[:, 1] > half_l)
    num_out_of_domain = (ood_x | ood_y).sum()
    num_inside_domain = N - num_out_of_domain
    assert num_out_of_domain > 0  # ensure we actually have out-of-bound points

    agg = PointAggregator(grid, rand_coords, warn_out_of_bounds=False)
    assert agg.cell_aggregates.sum() == num_inside_domain


def test_example_cell_counts_no_weights():
    point_coords = np.array([
        [0.1, 0.9],
        [0.3, 0.1],
        [0.6, 0.4],
        [0.76, 0.51],
        [0.99, 0.74],
    ])
    expected_counts = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 2],
        [1, 0, 0, 0]]
    )

    grid = SquareGridLayout(1, grid_center=(0.5, 0.5), num_cells=4 ** 2)  # 4x4 raster
    agg = PointAggregator(grid, point_coords)
    assert np.all(agg.cell_aggregates == expected_counts)


def test_example_cell_counts_with_weights():
    point_coords = np.array([
        [0.1, 0.9],
        [0.3, 0.1],
        [0.6, 0.4],
        [0.76, 0.51],
        [0.99, 0.74],
    ])
    point_weights = np.array([5, 4, 3, 2, 1])

    expected_weighted_sums = np.array([
        [0, 4, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 3],
        [5, 0, 0, 0]],
        dtype='float'
    )

    grid = SquareGridLayout(1, grid_center=(0.5, 0.5), num_cells=4 ** 2)  # 4x4 raster
    agg = PointAggregator(grid, point_coords, point_weights=point_weights)
    assert np.all(agg.cell_aggregates == expected_weighted_sums)
