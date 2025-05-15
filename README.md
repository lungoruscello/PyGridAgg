# PyGridAgg <img src="pygridagg/assets/icon.png" alt="icon" width="60" height="60"/>

[![PyPI Latest Release](https://img.shields.io/pypi/v/PyGridAgg.svg)](https://pypi.org/project/PyGridAgg/)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/lungoruscello/PyGridAgg/blob/master/LICENSE.txt)


**PyGridAgg** is a lightweight Python package that allows you to easily aggregate point data on 2D grids.
It includes efficient built-in aggregation schemes that can process large point datasets 
[quickly](#simple-and-fast). Defining grid layouts is also simple through several alternative 
[grid constructors](#defining-grid-layouts). While originally developed for geo-data analysis, 
PyGridAgg only depends on numpy and requires no GIS toolchain.

## Installation

**PyGridAgg** is available on [PyPI](https://pypi.org/project/PyGridAgg/) and can be
installed using `pip`:

`pip install pygridagg`

## Quickstart

```python
import matplotlib.pyplot as plt

import pygridagg as pga
from pygridagg.examples import load_japanese_earthquake_data

# Load example data on earthquakes around Japan
quake_coords, magnitudes = load_japanese_earthquake_data()

# Define a square grid layout with 10k cells encompassing all earthquake locations
layout = pga.SquareGridLayout.from_points(quake_coords, num_cells=100**2)

# Count earthquakes across grid cells
agg_counts = pga.CountAggregator(layout, quake_coords)

# Show a heatmap
agg_counts.plot(title="Earthquakes around Japan (2010-2023)")
plt.show()
```

## Simple and fast


For performance, all built-in [point aggregators](#built-in-point-aggregators) use [
`np.ufunc.at`](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html) for efficient in-place operations. 
In the timed example below, 10 million random points are aggregated on a grid with 250,000 cells. 
For illustration, points are aggregated using a weighted average, with point weights being assigned 
as a function of position:

```python
import time
import numpy as np
import matplotlib.pyplot as plt

import pygridagg as pga

# Define a grid layout on the unit square
bbox = 0, 1, 0, 1  # (x_min, x_max, y_min, y_max)
layout = pga.SquareGridLayout(*bbox, num_cells=500 ** 2)

# Generate random points, assign point weights in a smooth, periodic pattern 
N, freq = 10_000_000, 50
rand_coords = np.random.randn(N, 2) * 0.1 + 0.5
rand_weights = np.sin(freq * rand_coords[:, 0]) * np.cos(freq * rand_coords[:, 1])

# Time the data aggregation
start_time = time.time()
agg = pga.WeightedAverageAggregator(
    layout, rand_coords,
    point_weights=rand_weights,
)
elapsed_time = time.time() - start_time
print(f"Execution time: {elapsed_time:.f} seconds")

# Show a heatmap
agg.plot()
plt.show()
```

## Further details

### Defining grid layouts

You can choose between two different grid layouts:

* **SquareGridLayout**: Is restricted to have the same width and height, as well as the same number of columns and rows.

* **FlexibleGridLayout**: Allows you to independently set the grid's width and height, as well as the number of columns and rows.

When defining either grid layout, you can set the grid bounds by passing any of the following:

* a bounding box (via the default `__init__` of both layout classes);
* the desired centre coordinate and side dimensions of the grid (using the `from_centroid` constructor);
* a collection of template points from which grid limits are inferred (using the `from_points` constructor).

### Built-in point aggregators

The following aggregator classes are currently available:

* **CountAggregator**: Simply counts the number of points in each grid cell.

* **WeightedSumAggregator** and **WeightedAverageAggregator**:
  Compute a weighted sum or weighted average of points in each cell (given an array of aggregation weights).

* **MinimumWeightAggregator** and **MaximumWeightAggregator**:
  Compute the minimum or maximum weight of points in each grid cell (given an array of aggregation weights).


### Out-of-bounds points

Points outside the grid bounds do not affect the data aggregation. However, aggregator
classes will issue a warning when out-of-bounds points are present. To silence this warning,
set `warn_out_of_bounds=True` when instantiating an aggregator class.

### Column and row indexes

To access the column and row indexes of points, use the `grid_col_ids` and `row_col_ids` attributes
of an aggregator instance. Points located outside the grid bounds receive a column and row index of -1.

### Coordinate Reference Systems?

PyGridAgg aims to be as lightweight as possible and does not depend on GIS libraries like [
`pyproj`](https://pyproj4.github.io/pyproj/stable/)
or [`geopandas`](https://geopandas.org/en/stable/). As such, you need to handle transformations between coordinate
reference systems yourself. The package performs no cheks to see whether provided coordinates for points and grid layouts are valid.

### Implementing custom aggregators

You can define your own data aggregators by inheriting from `BasePointAggregator` and implementing the `aggregate` function.
The example below illustrates this with a custom aggregator class that only counts points inside a grid cell if an associated 
point weight is above a threshold value.

```python
import numpy as np

import pygridagg as pga
from pygridagg.examples import load_japanese_earthquake_data


class CustomThresholdCounter(pga.BasePointAggregator):
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
layout = pga.SquareGridLayout.from_points(quake_coords, num_cells=2_500)

# Only count earthquakes above magnitude 6
thresh = 6
agg = CustomThresholdCounter(layout, quake_coords, point_weights=magnitudes, threshold=thresh)

# Check that no earthquakes were 'lost'
assert agg.cell_aggregates.sum() == (magnitudes > thresh).sum()

# Show counts of major earthquakes with a heatmap
ax = agg.plot()
```

## Requirements

* `numpy`
* `matplotlib`

## License

This project is licensed under the MIT License. See [LICENSE.txt](https://github.com/lungoruscello/PyGridAgg/blob/master/LICENSE.txt) for details.
