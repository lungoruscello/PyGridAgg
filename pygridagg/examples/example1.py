import matplotlib.pyplot as plt

import pygridagg as pga
from pygridagg.examples import load_japanese_earthquake_data

# Load example data on earthquakes around Japan
quake_coords, magnitudes = load_japanese_earthquake_data()

# Define a square grid layout with 10k cells. The `from_points`
# constructor adjusts grid bounds to encompass all earthquake locations.
layout = pga.SquareGridLayout.from_points(quake_coords, num_cells=100**2)

# Compute the number of earthquakes and the maximum earthquake
# magnitude for all grid cells
agg_counts = pga.CountAggregator(layout, quake_coords)
agg_max_mag = pga.MaximumWeightAggregator(
    layout, quake_coords,
    point_weights=magnitudes
)

# Plot the data aggregates using heatmaps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
agg_counts.plot(ax=ax1, title="Earthquake count")
agg_max_mag.plot(ax=ax2, title="Largest magnitude", cmap='inferno')
plt.show()
