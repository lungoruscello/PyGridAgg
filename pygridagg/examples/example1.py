import matplotlib.pyplot as plt

from aggregate import SquareGridLayout, CountAggregator
from examples import load_japanese_earthquake_data

# Load example data on earthquakes around Japan
quake_coords, magnitudes = load_japanese_earthquake_data()

# Define a square grid layout with 2,500 cells, encompassing all
# earthquake locations
layout = SquareGridLayout.from_points(quake_coords, num_cells=2_500)

# Quickly count the number of earthquakes across grid cells
agg = CountAggregator(layout, quake_coords)

# Plot the aggregated data as a heatmap
ax = agg.plot()
plt.show()
