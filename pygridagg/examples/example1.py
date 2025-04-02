import matplotlib.pyplot as plt

from pygridagg import SquareGridLayout, PointAggregator
from pygridagg.examples import load_japanese_earthquake_data

# Load example data on earthquakes (geo-coordinates and magnitudes)
quake_coords, magnitudes = load_japanese_earthquake_data()

# Define a square grid layout with 2,500 cells that covers the
# spatial extent of the earthquake data
layout = SquareGridLayout.from_points(quake_coords, num_cells=2_500)

# Create a PointAggregator to count up earthquakes across grid cells
agg = PointAggregator(layout, quake_coords)

# Plot the aggregated data as a heatmap
ax = agg.plot()
plt.show()
