import time

import matplotlib.pyplot as plt
import numpy as np

import pygridagg as pga

# Define a grid layout on the unit square
bbox = 0, 1, 0, 1  # (x_min, x_max, y_min, y_max)
layout = pga.SquareGridLayout(*bbox, num_cells=500 ** 2)

# Generate random points
N = 10_000_000
rand_coords = np.random.randn(N, 2) * 0.1 + 0.5

# Assign point weights in a smooth, periodic pattern
freq = 50
rand_weights = np.sin(freq * rand_coords[:, 0]) * np.cos(freq * rand_coords[:, 1])

# Time the data aggregation
start_time = time.time()
agg = pga.WeightedAverageAggregator(
    layout, rand_coords,
    point_weights=rand_weights,
    warn_out_of_bounds=False
)
elapsed_time = time.time() - start_time

print(f"Execution time: {elapsed_time:.3f} seconds")

# Show a heatmap
agg.plot()
plt.show()
