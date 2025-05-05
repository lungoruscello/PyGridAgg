# PyGridAgg <img src="pygridagg/assets/icon.png" alt="icon" width="60" height="60"/> 

[![PyPI Latest Release](https://img.shields.io/pypi/v/PyGridAgg.svg)](https://pypi.org/project/PyGridAgg/)
[![License](https://img.shields.io/pypi/l/PyGridAgg.svg)](https://github.com/lungoruscello/PyGridAgg/blob/master/LICENSE.txt)

## About

PyGridAgg allows you to easily aggregate point data on 2D (spatial) grids. 
It includes efficient built-in aggregation schemes that can [process large point datasets quickly](#simple-and-fast). 
Defining grid layouts is also simple through several alternative grid constructors.

While originally developed for applications in geo-spatial data analysis, 
PyGridAgg only depends on `numpy` and requires no GIS toolchain.  


## Installation

You can install `pygridagg` using pip:

`pip install pygridagg`

## Quickstart

```python
import matplotlib.pyplot as plt
from pygridagg import *
from pygridagg.examples import load_japanese_earthquake_data

# Load example data on earthquakes around Japan
quake_coords, magnitudes = load_japanese_earthquake_data()

# Define a square grid that encompasses all earthquake locations
layout = SquareGridLayout.from_points(quake_coords, num_cells=2_500)

# Compute the number of quakes and the largest observed magnitude for all grid cells
agg_counts = CountAggregator(layout, quake_coords)
agg_max_mag = MaximumWeightAggregator(layout, quake_coords, point_weights=magnitudes)

# Plot the aggregated data using heatmaps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
agg_counts.plot(ax=ax1, title="No. of earthquakes")
agg_max_mag.plot(ax=ax2, title="Max. magnitude", cmap='magma')
plt.show()
```

## Simple and fast

All [built-in data aggregators](#built-in-data-aggregators) use [`np.ufunc.at`](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html) 
for fast in-place operations.

In the timed example below, 10 million random points are aggregated on a grid 
with 250,000 cells. For illustration, points are averaged using weights that
smoothly vary with position:

```python
import time
import numpy as np

from pygridagg.aggregate import SquareGridLayout, WeightedAverageAggregator

# Define a grid layout on the unit square
layout = SquareGridLayout.from_unit_square(num_cells=500 ** 2)

# Generate random points, assign weights in a smooth, periodic pattern 
N, freq = 10_000_000, 30
rand_coords = np.random.randn(N, 2) * 0.1 + 0.5
rand_weights = np.sin(freq * rand_coords[:, 0]) * np.cos(freq * rand_coords[:, 1])

# Time the data aggregation
start_time = time.time()
agg = WeightedAverageAggregator(
    layout, rand_coords,
    point_weights=rand_weights,
)
elapsed_time = time.time() - start_time
print(f"Execution time: {elapsed_time:.f} seconds")

# Show the result
agg.plot()
```

## Defining Grid Layouts

You can choose between two different grid layouts: 

* **SquareGridLayout**: Is restricted to have the same width and height as well as the same number of columns and rows. 

* **FlexibleGridLayout**: Allows you to set the overall width and height and the number of columns and rows independently 
of each other.

When instantiating either layout type, you can set the grid bounds in three different ways: 

1. by specifying a **bounding box** (using the default `__init__`  of both layout classes); 
2. by specifying the **centre coordinate and side dimensions** of the grid (using the `from_centroid` constructor);
3. by providing **template points** whose bounding box is used to infer the appropriate grid limits
(using the `from_points` constructor).   


## Built-in Point Aggregators

The following data aggregators are currently available: 

* **CountAggregator**: Simply counts the number of points in each grid cell.

* **WeightedSumAggregator** and **WeightedAverageAggregator**: 
Compute a weighted sum or weighted average of points in each cell.

* **MinimumWeightAggregator** and **MaximumWeightAggregator**: 
Compute the minimum or maximum weight of points in each grid cell.

All aggregators except the simple `CountAggregator` require an array of aggregation weights for all points. 

## Custom Aggregators

PyGridAgg allows you to define your own data aggregators by inheriting 
from `BasePointAggregator` and implementing the `aggregate` function.

The example below implements a custom aggregator that only counts points 
inside a grid cell if an associated point weight is above a certain 
*threshold*.

```python
import numpy as np

from pygridagg.aggregate import BasePointAggregator, SquareGridLayout
from pygridagg.examples import load_japanese_earthquake_data


class CustomThresholdCounter(BasePointAggregator):
    """Counts the number of points whose weight is above a threshold."""

    def aggregate(self, point_weights, threshold):
        # Initialise grid counts with zeroes
        counts = np.full(self.layout.shape, fill_value=0, dtype=int)

        # Select the column and row indexes of eligible points.
        # `self.inside_mask` is True for points inside the grid bounds.
        point_mask = self.inside_mask & (point_weights > threshold)
        col_ids = self.grid_col_ids[point_mask]
        row_ids = self.grid_row_ids[point_mask]

        # Use `np.add.at` for fast in-place addition
        np.add.at(counts, (row_ids, col_ids), 1)
        
        # Note: Returned array must always have shape (rows, columns)
        return counts  


quake_coords, magnitudes = load_japanese_earthquake_data()
layout = SquareGridLayout.from_points(quake_coords, num_cells=2_500)

# Only count earthquakes above magnitude 6
thresh = 6
agg = CustomThresholdCounter(layout, quake_coords, point_weights=magnitudes, threshold=thresh)

# Check that no earthquakes were 'lost'
assert agg.cell_aggregates.sum() == (magnitudes > thresh).sum()

# Plot counts of major earthquakes as a heatmap
ax = agg.plot()
```


## Further details

### Out-of-bounds points

Points located outside the grid bounds do not affect the data aggregation. However, aggregator 
classes will issue a warning when out-of-bounds points are present. To silence this warning, 
set `warn_out_of_bounds=True` when instantiating an aggregator class.


### Points' column and row indexes

You can access the column and row indexes of points using the `grid_col_ids` 
and `row_col_ids` attributes of an aggregator instance. Points located outside the grid bounds 
receive a column and row index of -1.

### Coordinate Reference Systems

PyGridAgg aims to be as lightweight as possible. It does not depend on GIS libraries 
like `pyproj` or `geopandas`. As such, users need to handle transformations between coordinate
reference systems themselves. No sanity checks are performed on user-provided coordinates.


## Requirements

* `numpy`
* `matplotlib`

## License

This project is licensed under the MIT License. See [LICENSE.txt](LICENSE.txt)  for details.
