import time

import numpy as np

from pygridagg import SquareGridLayout, PointAggregator

# Define a grid layout
side_length = 10
num_cells = 500 ** 2
layout = SquareGridLayout(
    side_length,
    grid_center=(0, 0),
    num_cells=num_cells
)

# Generate random points
N = 10_000_000
rand_coords = np.random.randn(N, 2)

# Time the data aggregation
start_time = time.time()
agg = PointAggregator(
    layout,
    rand_coords,
    warn_out_of_bounds=False  # set to True to issue warning for out-of-bounds points
)
num_inside_grid = agg.inside_mask.sum()  # no. of points inside the grid bounds
assert agg.cell_aggregates.sum() == num_inside_grid
elapsed_time = time.time() - start_time

print(f"Execution time: {elapsed_time:.4f} seconds")
