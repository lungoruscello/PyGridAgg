# PyGridAgg ![](https://github.com/lungoruscello/PyGridAgg/tree/master/pygridagg/assets/icon1.png)

[![PyPI Latest Release](https://img.shields.io/pypi/v/PyGridAgg.svg)](https://pypi.org/project/CryptNumPy/)
[![License](https://img.shields.io/pypi/l/PyGridAgg.svg)](https://github.com/lungoruscello/CryptNumPy/blob/master/LICENSE.txt)

## About

PyGridAgg allows you to quickly and easily aggregate point data on spatial grids.
It uses vectorised numpy operations [for speed](#Simple-but-fast) and scales to very
large point datasets.

## Installation

You can install `pygridagg` with pip:

`pip install pygridagg`

## Quickstart

```python
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
agg.plot()
```

### Explanation of the Quickstart:

**Grid Layout**:
The above example uses a `SquareGridLayout` in which the total width and height
are the same and the number of columns equals the number of rows. A more general `FlexibleGridLayout`
is also available (see below [TODO]).

**Auto-sizing**:
`from_points` automatically sets the overall grid size based on the spatial extent of provided point data.
The grid's spatial domain is then sub-divided into the requested number of cells.

**Point aggregation**:
`PointAggregator` handles the aggregation of all point data. By default, it simply counts
the number of points in each grid cell, but users can also obtain **weighted sums** as shown further below [TODO].

**Plotting**:
The `plot` method is only available when `matplotlib` is installed (optional dependency) 
and will  generate a heatmap of the aggregated data. 

## Simple but fast

PyGridAgg performs a simple task, but does so efficiently:

In the example below, 10 million random points are counted-up on a grid with 250,000 cells:

```python
import time
import numpy as np

from pygridagg import SquareGridLayout, PointAggregator

# Define a grid layout
side_length = 10
num_cells = 500**2
layout = SquareGridLayout(
    side_length, 
    grid_center=(0, 0), 
    num_cells=num_cells
)


# Generate random points
N = 10_000_000
rand_coords = np.random.randn(N, 2)

# Time the data aggregation
start_time = time.time()
agg = PointAggregator(
    layout, 
    rand_coords, 
    warn_out_of_bounds=False  # set to True to issue warning for out-of-bounds points
)
num_inside_grid = agg.inside_mask.sum()  # no. of points inside the grid bounds
assert agg.cell_aggregates.sum() == num_inside_grid
elapsed_time = time.time() - start_time

print(f"Execution time: {elapsed_time:.4f} seconds")
```

## Further functions

### Weighted sums

By default, each point will have weight 1 during aggregation. You can easily change this
by passing an array of weights to `PointAggregator`.

Using the earthquake example from earlier for simplicity, you can sum up earthquake 
*magnitudes* across grid cells as follows:

```python
from pygridagg import SquareGridLayout, PointAggregator
from pygridagg.examples import load_japanese_earthquake_data

quake_coords, magnitudes = load_japanese_earthquake_data()
layout = SquareGridLayout.from_points(quake_coords, num_cells=2_500)

agg = PointAggregator(layout, quake_coords, point_weights=magnitudes)
agg.plot(cmap='magma')
```


### Working with points' grid coordinates
Many users will be  

...


### Out-of-bounds points

...


### Different grid layouts

...

## Requirements

* `numpy`
* `matplotlib` (optional)

## License

This project is licensed under the [MIT License](LICENSE.txt).
