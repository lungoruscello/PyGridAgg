import time

import matplotlib.pyplot as plt
import numpy as np

from pygridagg import SquareGridLayout, WeightedAverageAggregator

# Define a grid layout
side_length = 10
num_cells = 500 ** 2
layout = SquareGridLayout(side_length, grid_center=(0, 0), num_cells=num_cells)

# Generate random points
N = 10_000_000
rand_coords = np.random.randn(N, 2)

# Assign point weights in a smooth, periodic pattern
freq = 5
rand_weights = np.sin(freq * rand_coords[:, 0]) * np.cos(freq * rand_coords[:, 1])

# Time the data aggregation
start_time = time.time()
agg = WeightedAverageAggregator(
    layout, rand_coords,
    point_weights=rand_weights,
    warn_out_of_bounds=False
)
elapsed_time = time.time() - start_time

print(f"Execution time: {elapsed_time:.4f} seconds")

# Show the result
agg.plot()
plt.show()
