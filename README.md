# PyGridAgg <img src="pygridagg/assets/icon.png" alt="icon" width="60" height="60"/> 

[![PyPI Latest Release](https://img.shields.io/pypi/v/PyGridAgg.svg)](https://pypi.org/project/CryptNumPy/)
[![License](https://img.shields.io/pypi/l/PyGridAgg.svg)](https://github.com/lungoruscello/CryptNumPy/blob/master/LICENSE.txt)

## About

PyGridAgg lets you quickly aggregate point data on spatial grids. 
It includes built-in aggregation schemes that are designed to aggregate 
[large point datasets quickly](#Simple-but-fast) 
through efficient numpy operations. Defining grid layouts is also 
simple through several alternative constructors.

While originally developed for applications in geo-spatial data analysis, 
PyGridAgg only depends on `numpy` and requires no GIS toolchain.  


## Installation

You can install `pygridagg` with pip:

`pip install pygridagg`

## Quickstart

### Simple counts over grid cells 

```python
from pygridagg import SquareGridLayout, CountAggregator
from pygridagg.examples import load_japanese_earthquake_data

# Load example data on earthquakes around Japan
quake_coords, magnitudes = load_japanese_earthquake_data()

# Define a square grid layout with 2,500 cells, encompassing all
# earthquake locations
layout = SquareGridLayout.from_points(quake_coords, num_cells=2_500)

# Quickly count the number of earthquakes across grid cells
agg = CountAggregator(layout, quake_coords)

# Plot the aggregated data as a heatmap
ax = agg.plot()
```

### Weighted averages over grid cells

```python
from pygridagg import SquareGridLayout, WeightedAverageAggregator
from pygridagg.examples import load_japanese_earthquake_data

# Load example data on earthquakes around Japan
quake_coords, magnitudes = load_japanese_earthquake_data()

# Define a square grid layout with 2,500 cells, encompassing all
# earthquake locations
layout = SquareGridLayout.from_points(quake_coords, num_cells=2_500)

# Quickly compute average earthquake magnitudes across grid cells
agg = WeightedAverageAggregator(layout, quake_coords, point_weights=magnitudes)

# Plot the aggregated data as a heatmap
ax = agg.plot()
```

### Explanation of the Quickstart examples:

**Grid Layout**:
Both quickstart examples use a `SquareGridLayout`. This defines a *restricted* grid 
with equal width and height and an equal number of columns and rows. A more 
general `FlexibleGridLayout` is also available.

**Auto-sizing**:
`from_points` automatically adjusts the grid bounds to encompass all provided 
points, then divides the spatial domain into the specified number of grid cells.
Alternatively, grid layouts [can also be defined](TODO) from bounding boxes, as well as from centre coordinates 
and side dimensions.

**Point aggregation**:
The two quickstart examples use the `CountAggregator` and `WeightedAverageAggregator`
to respectively compute the number and average magnitude of earthquakes within grid 
cells. Three additional [point aggregators](#built-in-data-aggregators) are 
available and users can implement their own via [subclassing](#customisation).   
 
**Plotting**:
The `plot` method is only available when `matplotlib` is installed (optional dependency) 
and will generate a heatmap of the aggregated data. 


## Simple but fast

PyGrid performs simple tasks, but does so efficiently. 

In the timed example below, 10 million random points are aggregated on a grid 
with 250,000 cells. For illustration, points are averaged using weights that
smoothly vary with position:

```python
import time
import numpy as np

from aggregate import SquareGridLayout, WeightedAverageAggregator

# Define a grid on the unit square
layout = SquareGridLayout.from_unit_square(num_cells=500 ** 2)

# Generate random points
N = 10_000_000
rand_coords = np.random.randn(N, 2) * 0.1 + 0.5

# Assign point weights in a smooth, periodic pattern
freq = 30
rand_weights = np.sin(freq * rand_coords[:, 0]) * np.cos(freq * rand_coords[:, 1])

# Time the data aggregation
start_time = time.time()
agg = WeightedAverageAggregator(
    layout, rand_coords,
    point_weights=rand_weights,
)
elapsed_time = time.time() - start_time

print(f"Execution time: {elapsed_time:.4f} seconds")

# Show the result
agg.plot()
```
 
## Built-in Data Aggregators

Built-in data aggregators are listed below. All of them use [`np.ufunc.at`](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html) 
for fast inplace point-data aggregations.

**CountAggregator**: Simply counts the number of points in each grid cell.

**WeightedSumAggregator** and **WeightedAverageAggregator**: 
Respectively compute a weighted sum and a weighted average of points in each cell (given an array of 
aggregation weights for all points).

**MinimumWeightAggregator** and **MaximumWeightAggregator**: 
Respectively compute the minimum 
and maximum weight of points in each grid cell (given an array of 
aggregation weights for all points). 

## Customisation

PyGridAgg allows you to define your own data aggregators. To do so, inherit 
from `BasePointAggregator` and implement the `aggregate` function, which is expected 
to return a 2D numpy array.

The example below shows a simple, custom aggregator that counts points whose weight 
is above a specified threshold.

```python
import numpy as np

from aggregate import BasePointAggregator, SquareGridLayout
from examples import load_japanese_earthquake_data


class CustomThresholdCounter(BasePointAggregator):
    """Counts the number of points whose weight is above a threshold."""

    def aggregate(self, point_weights, threshold):
        # Initialise grid counts with zeroes. `self.layout.shape`
        # gives the grid size in terms of (rows, columns).
        counts = np.full(self.layout.shape, fill_value=0, dtype=int)

        # Select the column and row indexes of eligible points.
        # `self.inside_mask` is True for points inside the grid bounds.
        point_mask = self.inside_mask & (point_weights > threshold)
        col_ids = self.grid_col_ids[point_mask]
        row_ids = self.grid_row_ids[point_mask]

        # Use `np.add.at` for fast in-place addition
        np.add.at(counts, (row_ids, col_ids), 1)  # noqa
        
        # Note: Return value must always have shape (rows, columns)
        return counts  


# Load example data on earthquakes around Japan
quake_coords, magnitudes = load_japanese_earthquake_data()

# Define a square grid layout with 2,500 cells, encompassing all
# earthquake locations
layout = SquareGridLayout.from_points(quake_coords, num_cells=2_500)

# Quickly count earthquakes above magnitude 6 across grid cells
thresh = 6
agg = CustomThresholdCounter(layout, quake_coords, point_weights=magnitudes, threshold=thresh)

# Check that no earthquakes were 'lost'
assert agg.cell_aggregates.sum() == (magnitudes > thresh).sum()

# Plot counts of major earthquakes as a heatmap
ax = agg.plot()
```

## Defining Grid Layouts

... 

##

## Further details

### Warning for out-of-bounds points

Points located outside the grid bounds do not affect the data aggregation.
Nonetheless, you will be shown a warning when passing out-of-bounds points 
to an aggregator. You can silence this warning as follows: 


...

### Useful aggregator attributes

...


### What about Coordinate Reference Systems?

PyGridAgg is lightweight and does not depend on GIS libraries, such as `pyproj` or 
`geopandas`. This reduces dependencies, but also means that no checks are 
performed on coordinate reference systems – which users need to handle themselves.  



## Requirements

* `numpy`
* `matplotlib` (optional)

## License

This project is licensed under the [MIT License](LICENSE.txt).
