import numpy as np
import pytest

from pygridagg.aggregate import (
    CountAggregator,
    WeightedSumAggregator,
    WeightedAverageAggregator,
    MinimumWeightAggregator,
    MaximumWeightAggregator,
    PointLocaliser
)
from pygridagg.examples import load_japanese_earthquake_data
from pygridagg.grid_layouts import SquareGridLayout

np.random.seed(42)

# TODO: Add all-OOB test
# TODO: Add test for point weights with OOB

def test_empty_point_data_raises():
    empty_coords = np.array([]).reshape(0, 2)
    layout = SquareGridLayout.from_centroid(
        10, grid_center=(0, 0), num_cells=5 ** 2
    )
    with pytest.raises(ValueError):
        CountAggregator(layout, empty_coords)


def test_missing_point_weights_raises():
    any_coords = np.random.randn(3, 2)
    layout = SquareGridLayout.from_points(any_coords, num_cells=9)
    with pytest.raises(TypeError):
        WeightedSumAggregator(layout, any_coords)


def test_localisation_only():
    N = 10
    rand_coords = np.random.randn(N, 2)
    layout = SquareGridLayout.from_centroid(
        1, grid_center=(0.5, 0.5), num_cells=5 ** 2
    )
    agg = PointLocaliser(layout, rand_coords)

    assert agg.cell_aggregates is None
    assert agg.grid_col_ids.shape == (N,)
    assert agg.grid_row_ids.shape == (N,)
    assert agg.inside_mask.shape == (N,)


def test_no_points_lost_no_oob_quakes():
    coords, _ = load_japanese_earthquake_data()
    N = coords.shape[0]

    layout = SquareGridLayout.from_points(
        coords,
        padding_percent=0.001,
        num_cells=20 ** 2
    )

    agg = CountAggregator(layout, coords)
    assert agg.cell_aggregates.sum() == N


def test_no_points_lost_with_oob_quakes():
    coords, _ = load_japanese_earthquake_data()
    N = coords.shape[0]

    # Below, we construct a square grid centred on Tokyo
    # while the full earthquake data extends much further
    # into the Japanese archipelago.
    # Note: Coordinates are in degrees lon/lat.

    tokyo = 139.753481, 35.684568
    layout = SquareGridLayout.from_centroid(
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

    agg = CountAggregator(layout, coords, warn_out_of_bounds=False)
    assert agg.cell_aggregates.sum() == num_inside_domain


def test_bespoke_example1_counts(
        bespoke_test_points1,
        bespoke_test_layout1,
):
    expected_counts = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 2],
        [2, 0, 0, 0]]
    )

    agg = CountAggregator(bespoke_test_layout1, bespoke_test_points1)
    assert np.all(agg.cell_aggregates == expected_counts)


def test_bespoke_example1_weighted_sums(bespoke_agg_args1):
    # fill value zero
    expected_weighted_sums = np.array([
        [0, 4, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 3],
        [4, 0, 0, 0]],
        dtype='float'
    )
    agg = WeightedSumAggregator(fill_value=0, **bespoke_agg_args1)
    assert np.all(agg.cell_aggregates == expected_weighted_sums)

    # fill value NaN
    expected_weighted_sums[expected_weighted_sums == 0] = np.nan
    agg = WeightedSumAggregator(fill_value=np.nan, **bespoke_agg_args1)
    assert np.allclose(agg.cell_aggregates, expected_weighted_sums, equal_nan=True)


def test_bespoke_example1_weighted_averages(bespoke_agg_args1):
    # fill value zero
    expected_weighted_avgs = np.array([
        [0, 4, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 3 / 2],
        [4 / 2, 0, 0, 0]],
        dtype='float'
    )
    agg = WeightedAverageAggregator(fill_value=0, **bespoke_agg_args1)
    assert np.all(agg.cell_aggregates == expected_weighted_avgs)

    # fill value NaN
    expected_weighted_avgs[expected_weighted_avgs == 0] = np.nan
    agg = WeightedAverageAggregator(fill_value=np.nan, **bespoke_agg_args1)
    assert np.allclose(agg.cell_aggregates, expected_weighted_avgs, equal_nan=True)


def test_bespoke_example1_weight_minima(bespoke_agg_args1):
    expected_minima = np.array([
        [0, 4, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0]],
        dtype='float'
    )
    # replace zeroes with NaNs (fill value)
    expected_minima[expected_minima == 0] = np.nan

    agg = MinimumWeightAggregator(fill_value=np.nan, **bespoke_agg_args1)
    assert np.allclose(agg.cell_aggregates, expected_minima, equal_nan=True)


def test_bespoke_example1_weight_maxima(bespoke_agg_args1):
    expected_maxima = np.array([
        [0, 4, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 2],
        [5, 0, 0, 0]],
        dtype='float'
    )
    # replace zeroes with NaNs (fill value)
    expected_maxima[expected_maxima == 0] = np.nan

    agg = MaximumWeightAggregator(fill_value=np.nan, **bespoke_agg_args1)
    assert np.allclose(agg.cell_aggregates, expected_maxima, equal_nan=True)


def test_bespoke_example2_counts(
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

    agg = CountAggregator(bespoke_test_layout2, bespoke_test_points2)
    assert np.all(agg.cell_aggregates == expected_counts)


def test_bespoke_example2_weighted_sums(bespoke_agg_args2):
    # fill value zero
    expected_weighted_sums = np.array([
        [-1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, -3, 0]],
        dtype='float'
    )

    agg = WeightedSumAggregator(fill_value=0, **bespoke_agg_args2)
    assert np.all(agg.cell_aggregates == expected_weighted_sums)

    # fill value NaN
    expected_weighted_sums[expected_weighted_sums == 0] = np.nan
    agg = WeightedSumAggregator(fill_value=np.nan, **bespoke_agg_args2)
    assert np.allclose(agg.cell_aggregates, expected_weighted_sums, equal_nan=True)


def test_bespoke_example2_weighted_averages(bespoke_agg_args2):
    # fill value zero
    expected_weighted_avgs = np.array([
        [-1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1 / 2, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, -3, 0]],
        dtype='float'
    )

    agg = WeightedAverageAggregator(fill_value=0, **bespoke_agg_args2)
    assert np.all(agg.cell_aggregates == expected_weighted_avgs)

    # fill value NaN
    expected_weighted_avgs[expected_weighted_avgs == 0] = np.nan
    agg = WeightedAverageAggregator(fill_value=np.nan, **bespoke_agg_args2)
    assert np.allclose(agg.cell_aggregates, expected_weighted_avgs, equal_nan=True)


def test_bespoke_example2_weight_minima(bespoke_agg_args2):
    expected_minima =  np.array([
        [-1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -5, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, -3, 0]],
        dtype='float'
    )
    # replace zeroes with NaNs (fill value)
    expected_minima[expected_minima == 0] = np.nan

    agg = MinimumWeightAggregator(fill_value=np.nan, **bespoke_agg_args2)
    assert np.allclose(agg.cell_aggregates, expected_minima, equal_nan=True)


def test_bespoke_example2_weight_maxima(bespoke_agg_args2):
    expected_maxima = np.array([
        [-1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, -3, 0]],
        dtype='float'
    )
    # replace zeroes with NaNs (fill value)
    expected_maxima[expected_maxima == 0] = np.nan

    agg = MaximumWeightAggregator(fill_value=np.nan, **bespoke_agg_args2)
    assert np.allclose(agg.cell_aggregates, expected_maxima, equal_nan=True)


@pytest.fixture
def bespoke_test_layout1():
    return SquareGridLayout.from_centroid(
        1, grid_center=(0.5, 0.5), num_cells=16
    )


@pytest.fixture
def bespoke_test_layout2():
    return SquareGridLayout.from_centroid(
        2, grid_center=(-1, -1), num_cells=36
    )


@pytest.fixture
def bespoke_test_points1():
    # constructed assuming a 4x4 square grid on unit square
    return np.array([
        [0.10, 1.00],
        [0.11, 0.99],
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


@pytest.fixture
def bespoke_test_weights1():
    return np.array([5, -1, 4, 3, 2, 1])


@pytest.fixture
def bespoke_test_weights2():
    return np.array([-1, +2, -3, +4, -5])


@pytest.fixture
def bespoke_agg_args1(bespoke_test_layout1, bespoke_test_points1, bespoke_test_weights1):
    return dict(
        grid_layout=bespoke_test_layout1,
        points=bespoke_test_points1,
        point_weights=bespoke_test_weights1
    )


@pytest.fixture
def bespoke_agg_args2(bespoke_test_layout2, bespoke_test_points2, bespoke_test_weights2):
    return dict(
        grid_layout=bespoke_test_layout2,
        points=bespoke_test_points2,
        point_weights=bespoke_test_weights2
    )
