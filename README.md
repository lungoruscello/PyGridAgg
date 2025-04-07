# PyGridAgg <img src="pygridagg/assets/icon.png" alt="icon" width="60" height="60"/> 

[![PyPI Latest Release](https://img.shields.io/pypi/v/PyGridAgg.svg)](https://pypi.org/project/PyGridAgg/)
[![License](https://img.shields.io/pypi/l/PyGridAgg.svg)](https://github.com/lungoruscello/PyGridAgg/blob/master/LICENSE.txt)

## About

PyGridAgg lets you quickly aggregate point data on spatial grids. 
It includes built-in aggregation schemes that are designed to aggregate 
[large point datasets quickly](#Simple-but-fast) 
through efficient numpy operations. Defining grid layouts is also 
simple through several alternative constructors.

While originally developed for applications in geo-spatial data analysis, 
PyGridAgg only depends on `numpy` and requires no GIS toolchain.  


## Installation

You can install `pygridagg` using pip:

`pip install pygridagg`

## Quickstart

### Counting points within grid cells 

```python
from pygridagg import SquareGridLayout, CountAggregator
from pygridagg.examples import load_japanese_earthquake_data

# Load example data on earthquakes around Japan
quake_coords, magnitudes = load_japanese_earthquake_data()

# Define a square grid layout with 2,500 cells, encompassing all
# earthquake locations
layout = SquareGridLayout.from_points(quake_coords, num_cells=2_500)

# Quickly count the number of earthquakes within grid cells
agg = CountAggregator(layout, quake_coords)

# Plot the aggregated data as a heatmap
ax = agg.plot()
```

### Weighted averages

```python
from pygridagg import SquareGridLayout, WeightedAverageAggregator
from pygridagg.examples import load_japanese_earthquake_data

# Load example data on earthquakes around Japan
quake_coords, magnitudes = load_japanese_earthquake_data()

# Define a square grid layout with 2,500 cells, encompassing all
# earthquake locations
layout = SquareGridLayout.from_points(quake_coords, num_cells=2_500)

# Quickly compute average earthquake magnitudes within grid cells
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
Alternatively, grid layouts [can also be defined](#defining-grid-layouts) from bounding boxes, as well as from centre coordinates 
and side dimensions.

**Point aggregation**:
The quickstart examples respectively use the `CountAggregator` and `WeightedAverageAggregator`
to compute the number and average magnitude of earthquakes within grid 
cells. Three additional [point aggregators](#built-in-data-aggregators) are 
available and users can implement their own via subclassing.   
 
**Plotting**:
The `plot` method is only available when `matplotlib` is installed (optional dependency) 
and will generate a heatmap of the aggregated data. 


## Simple but fast

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
print(f"Execution time: {elapsed_time:.f} seconds")

# Show the result
agg.plot()
```
 
## Built-in Data Aggregators

The built-in data aggregators are listed below. All of them use [`np.ufunc.at`](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html) 
for fast inplace point-data aggregations.

* **CountAggregator**: Simply counts the number of points in each grid cell

 
* **WeightedSumAggregator** and **WeightedAverageAggregator**: 
Respectively compute a weighted sum and a weighted average of points in each cell (given an array of 
aggregation weights for all points).
 

* **MinimumWeightAggregator** and **MaximumWeightAggregator**: 
Respectively compute the minimum 
and maximum weight of points in each grid cell (given an array of 
aggregation weights for all points). 

## Custom Aggregators

PyGridAgg allows you to define your own data aggregators by inheriting 
from `BasePointAggregator` and implementing the `aggregate` function.

To illustrate, let's implement an aggregator that only counts points  
within grid cells if an associated point weight is above a specified 
*threshold*.

```python
import numpy as np

from aggregate import BasePointAggregator, SquareGridLayout
from examples import load_japanese_earthquake_data


class CustomThresholdCounter(BasePointAggregator):
    """Counts the number of points whose weight is above a threshold."""

    def aggregate(self, point_weights, threshold):
        # This method is expected to return a 2D numpy array
        # whose shape matches `self.layout.shape` (rows, columns)
        
        # Initialise grid counts with zeroes. 
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

# Quickly count earthquakes above magnitude 6 within grid cells
thresh = 6
agg = CustomThresholdCounter(layout, quake_coords, point_weights=magnitudes, threshold=thresh)

# Check that no earthquakes were 'lost'
assert agg.cell_aggregates.sum() == (magnitudes > thresh).sum()

# Plot counts of major earthquakes as a heatmap
ax = agg.plot()
```

## Defining Grid Layouts

PyGridAgg provides two different grid layouts: The **SquareGridLayout** is restricted
to have the same width and height as well as the same number of columns and rows. 
The **FlexibleGridLayout** allows you to set the overall width and height and  
number of columns and rows independently of each other.

For both layout types, you can set the *grid bounds* in three different ways: 

* Specify a **bounding box** (default).


* Specify a **centre coordinate and side dimensions** for the grid (using the 
`from_centroid` method).


* Provide **template points** whose bounding box is used to infer 
appropriate grids limits (using the `from_points` method).   


## Frequently Asked Questions

### How do I access the grid coordinates of points?

Aggregator classes store the column and row indexes of points using the `grid_col_ids` 
and `row_col_ids` attributes. Points located outside the grid limits are marked with 
a column and row index of -1.

### How are out-of-bounds points handled?

Points located outside the grid bounds do not affect the data aggregation.
However, built-in aggregator classes will issue a warning when out-of-bounds
points are present, unless you can silence this warning (`warn_out_of_bounds=True`).  


### What about Coordinate Reference Systems?

PyGridAgg aims to be as lightweight as possible and does not depend on GIS libraries, 
such as `pyproj` or `geopandas`. Users need to handle transformations between coordinate
reference systems themselves and PyGridAgg performs no checks to ensure that point 
locations and grid layouts are expressed in the same CRS.

## Requirements

* `numpy`
* `matplotlib` (optional)

## License

This project is licensed under the MIT License. See [LICENSE.txt](LICENSE.txt)  for details.
