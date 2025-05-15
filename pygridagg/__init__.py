__about__ = "Quickly and easily aggregate point data on customisable 2D grids."
__version__ = '0.1.0'
__url__ = "https://github.com/lungoruscello/PyGridAgg"
__license__ = "MIT"
__author__ = "S. Langenbach"


from .aggregate import (
    BasePointAggregator,
    CountAggregator,
    WeightedSumAggregator,
    WeightedAverageAggregator,
    MinimumWeightAggregator,
    MaximumWeightAggregator,
    PointLocaliser
)
from .grid_layouts import FlexibleGridLayout, SquareGridLayout
from .examples import load_japanese_earthquake_data
