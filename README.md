# PyGridAgg <img src="pygridagg/assets/icon.png" alt="icon" width="60" height="60"/> 

[![PyPI Latest Release](https://img.shields.io/pypi/v/PyGridAgg.svg)](https://pypi.org/project/CryptNumPy/)
[![License](https://img.shields.io/pypi/l/PyGridAgg.svg)](https://github.com/lungoruscello/CryptNumPy/blob/master/LICENSE.txt)

## About

PyGridAgg allows you to easily aggregate point data on spatial grids, with different
aggregation strategies and a quick definition of grid layouts. It leverages vectorised 
numpy operations and is designed to aggregate [large point datasets quickly](#Simple-but-fast).

While originally developed for applications in geo-spatial data analysis, PyGridAgg 
only depends on `numpy` and requires no GIS toolchain.  


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
The quickstart examples both use a `SquareGridLayout`. This is a *restricted* grid 
layout with an equal width and height and an equal number of columns and rows. 
A more general `FlexibleGridLayout` is also available (see further [below](TODO)).

**Auto-sizing**:
`from_points` automatically adjusts the grid bounds to encompass all provided 
points, then divides the spatial domain into the specified number of grid cells.

**Point aggregation**:
PyGridAgg supports [five built-in aggregation schemes](#built-in-data-aggregators). The quickstart examples 
respectively use the `CountAggregator` and `WeightedAverageAggregator` to quickly 
count earthquakes and compute average earthquake *magnitudes* within grid cells. 

**Plotting**:
The `plot` method is only available when `matplotlib` is installed (optional dependency) 
and will  generate a heatmap of the aggregated data. 


## Simple but fast


In the timed example below, 10 million random points are aggregated on a grid 
with 250,000 cells. Aggregation is performed using a weighted average:

```python
import time 
import numpy as np

from pygridagg import SquareGridLayout, WeightedAverageAggregator

# Define a grid layout
layout = SquareGridLayout(10, grid_center=(0, 0), num_cells=500**2)

# Generate random points
rand_coords = np.random.randn(10_000_000, 2)

# Assign weights to points in a smooth, periodic pattern
freq = 5
rand_weights = np.sin(freq * rand_coords[:, 0]) * np.cos(freq * rand_coords[:, 1])

# Time the data aggregation
start_time = time.time()
agg = WeightedAverageAggregator(
    layout, rand_coords, 
    point_weights=rand_weights,
    warn_out_of_bounds=False
)
elapsed_time = time.time() - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")

# Show the result
agg.plot()
```
 
## Built-in Data Aggregators

Available data aggregators are listed below. All of them use [`np.ufunc.at`](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html) for fast inplace aggregations.

**CountAggregator**: Simply counts the number of points in each grid cell.

**WeightedSumAggregator** and **WeightedAverageAggregator**: 
These classes respectively compute a weighted sum and weighted average of  points in each cell (given an array of 
aggregation weights for all points).

**MinimumWeightAggregator** and **MaximumWeightAggregator**: 
These classes respectively compute the minimum 
and maximum weight of points in each grid cell (given an array of 
aggregation weights for all points). 

## Customisation

You can define your own data aggregators by subclassing from `BasePointAggregator` and 
implementing the `aggregate` function. This function is expected to return a 2D numpy 
array whose shape matches the number of rows and columns in the grid. 

The example below illustrates this with a simple, custom aggregator that counts points
whose weight is above a specified threshold.

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

# Plot counts of major earthquakles as a heatmap
ax = agg.plot()
```

## Defining Grid Layouts

TODO

##

## Further details

### Extracting points' grid coordinates

...

### Out-of-bounds points

...


## Requirements

* `numpy`
* `matplotlib` (optional)

## License

This project is licensed under the [MIT License](LICENSE.txt).
