import matplotlib.pyplot as plt
import numpy as np

import pygridagg as pga
from pygridagg.examples import load_japanese_earthquake_data


# Implement a custom aggregator via subclassing

class CustomThresholdCounter(pga.BasePointAggregator):
    """Counts the number of points whose weight is above a threshold."""

    def aggregate(self, point_weights, threshold):
        # Initialise grid counts with zeroes. `self.layout.shape`
        # gives the grid size in terms of (rows, columns).
        counts = np.full(self.layout.shape, fill_value=0, dtype=int)

        # Select the column and row indexes of eligible points.
        # `self.inside_mask` is Boolean mask and is True for points
        # inside the grid bounds.
        point_mask = self.inside_mask & (point_weights > threshold)
        col_ids = self.grid_col_ids[point_mask]
        row_ids = self.grid_row_ids[point_mask]

        # Use `np.add.at` for fast in-place addition
        np.add.at(counts, (row_ids, col_ids), 1)  # noqa

        return counts


# Load example data on earthquakes around Japan
quake_coords, magnitudes = load_japanese_earthquake_data()

# Define a square grid layout with 2,500 cells, encompassing all
# earthquake locations
layout = pga.SquareGridLayout.from_points(quake_coords, num_cells=2_500)

# Quickly count earthquakes above magnitude 6 within grid cells
thresh = 6
agg = CustomThresholdCounter(layout, quake_coords, point_weights=magnitudes, threshold=thresh)

# Check that no earthquakes were 'lost'
assert agg.cell_aggregates.sum() == (magnitudes > thresh).sum()

# Show counts of major earthquakes with a heatmap
ax = agg.plot()
plt.show()
