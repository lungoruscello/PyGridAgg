# PyGridAgg <img src="pygridagg/assets/icon.png" alt="icon" width="60" height="60"/> 

[![PyPI Latest Release](https://img.shields.io/pypi/v/PyGridAgg.svg)](https://pypi.org/project/CryptNumPy/)
[![License](https://img.shields.io/pypi/l/PyGridAgg.svg)](https://github.com/lungoruscello/CryptNumPy/blob/master/LICENSE.txt)

## About

PyGridAgg allows you to quickly aggregate point data on spatial grids, with different
aggregation strategies and grid layouts. It leverages vectorised numpy operations [for speed](#Simple-but-fast) 
and is designed to handle large point datasets.

While originally developed for applications in geo-spatial data analysis, PyGridAgg 
only depends on `numpy` and requires no GIS toolchain.  


## Installation

You can install `pygridagg` with pip:

`pip install pygridagg`

## Quickstart

### Simple counts

```python
from pygridagg import SquareGridLayout, CountAggregator
from pygridagg.examples import load_japanese_earthquake_data

# Load example data on earthquakes near Japan
quake_coords, magnitudes = load_japanese_earthquake_data()

# Define a square grid layout with 2,500 cells, encompassing all
# earthquake locations
layout = SquareGridLayout.from_points(quake_coords, num_cells=2_500)

# Quickly count the number of earthquakes across grid cells
agg = CountAggregator(layout, quake_coords)

# Plot the aggregated data as a heatmap
ax = agg.plot()
```

### Weighted averages

```python
from pygridagg import SquareGridLayout, WeightedAverageAggregator
from pygridagg.examples import load_japanese_earthquake_data

# Load example data on earthquakes near Japan
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
The two examples above both use a `SquareGridLayout`. This is a *restricted* grid 
layout with an equal width and height and an equal number of columns and rows. 
A more general `FlexibleGridLayout` is also available (see [further  below](TODO)).

**Auto-sizing**:
`from_points` automatically adjusts the grid bounds to encompass all provided 
points, then divides the spatial domain into the specified number of grid cells.

**Point aggregation**:
PyGridAgg supports several different aggregation strategies. The quickstart examples 
employ the `CountAggregator`, which simply counts the number of points (here: earthquakes) 
across grid cells, as well as the `WeightedAverageAggregator`, which is used in the second 
example to compute average earthquake *magnitudes* across the grid. 
[Further aggregation strategies](TODO) are available, and users can implement their own 
aggregation schemes through subclassing.
   

**Plotting**:
The `plot` method is only available when `matplotlib` is installed (optional dependency) 
and will  generate a heatmap of the aggregated data. 


## Simple but fast

PyGridAgg performs simple tasks, but does so efficiently.

In the timed example below, 10 million random points are aggregated on a grid with 250,000 cells:   

```python
import time
import numpy as np

from pygridagg import SquareGridLayout, WeightedAverageAggregator

# Define a grid layout
side_length = 10
num_cells = 500**2
layout = SquareGridLayout(side_length, grid_center=(0, 0), num_cells=num_cells)

# Generate random points
N = 10_000_000
rand_coords = np.random.randn(N, 2)

# Assign point weights in a smooth, periodic pattern
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
 
## Further functions



### Extracting points' grid coordinates


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
