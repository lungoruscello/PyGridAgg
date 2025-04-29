import matplotlib.pyplot as plt
from pygridagg import *
from pygridagg.examples import load_japanese_earthquake_data

# Load example data on earthquakes around Japan
quake_coords, magnitudes = load_japanese_earthquake_data()

# Define a square grid layout with 2,500 cells. The `from_points`
# constructor adjusts grid bounds to encompass all earthquake locations.
layout = SquareGridLayout.from_points(quake_coords, num_cells=2_500)

# Compute the number of earthquakes and the maximum earthquake
# magnitude for all grid cells
agg_counts = CountAggregator(layout, quake_coords)
agg_max_mag = MaximumWeightAggregator(
    layout, quake_coords,
    point_weights=magnitudes
)

# Plot the aggregated data using heatmaps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
agg_counts.plot(ax=ax1, title="No. of earthquakes")
agg_max_mag.plot(ax=ax2, title="Max. magnitude", cmap='magma')
plt.show()
