"""
Fast aggregation of 2D points on spatial grids.
========================================================================
"""
import warnings
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from .grid_layouts import *  # noqa
from .utils import ensure_array_shape


# ***************************
# Point Aggregators
# ***************************
class BasePointAggregator(ABC):
    """
    Base class for grid-based point aggregators.

    Attributes
    ----------
    grid_col_ids : np.ndarray
        1D array with column indices for each point. Out-of-bounds points have index -1.
    grid_row_ids : np.ndarray
        1D array with row indices for each point. Out-of-bounds points have index -1.
    inside_mask : np.ndarray
        1D Boolean mask indicating whether each point is inside the grid bounds.
    N : int
        Number of points.
    N_inside : int
        Number of points inside the grid.
    cell_aggregates : np.ndarray
        2D array of aggregated point data for each grid cell, or None if no aggregation
        is performed.

    Methods
    -------
    aggregate(point_weights=None)
        Aggregate point data within grid cells according to the specific aggregation
        strategy that subclasses implemented.
    plot(ax=None, colorbar=True, colorbar_kwargs=None, **kwargs)
        Visualise point-data aggregates with a heatmap.
    """

    _check_implementation = True

    def __init__(
            self,
            grid_layout,
            points,
            *,
            warn_out_of_bounds=True,
            **agg_func_kwargs
    ):
        """
        Initialise a `PointAggregator` with the given grid layout and points.

        Parameters
        ----------
        grid_layout : Instance of `FlexibleGridLayout` or `SquareGridLayout`
            The spatial grid layout to be used when aggregating points.
        points : np.ndarray
            Array with shape (N, 2) holding the x and y coordinates for a collection
            of N points.
        warn_out_of_bounds : bool, optional
            Whether to show a warning for points that are out-of-bounds. Default is True.
            If set to False, no warnings for out-of-bounds points will be shown.
        **agg_func_kwargs
            Additional keyword arguments passed to the aggregation function.
        """
        self.layout = grid_layout
        self.warn_out_of_bounds = warn_out_of_bounds

        points = ensure_array_shape(points)
        self._assign_points_to_grid_cells(points)
        self.N = len(points)  # noqa
        self.N_inside = int(self.inside_mask.sum())  # noqa

        self.cell_aggregates = self.aggregate(**agg_func_kwargs)

        if self._check_implementation:
            # Perform a sanity check on whether subclasses have returned
            # data aggregates of the expected shape
            if self.cell_aggregates is None or self.cell_aggregates.shape != self.layout.shape:
                raise RuntimeError(
                    "Invalid implementation of `BasePointAggregator`. Please "
                    f"check the return value of the `aggregate` method "
                    f"implemented by `{self.__class__.__name__}`."
                )


    @abstractmethod
    def aggregate(self, *args, **kwargs):
        """
        Abstract method to aggregate point data within grid cells with a
        custom strategy. Subclasses must define how the aggregation occurs
        (e.g., counting or weighted sums).

        Parameters
        ----------
        *args, **kwargs : Arguments specific to the chosen aggregation
                          strategy, such as `point_weights` for weighted
                          aggregations (see Notes).

        Returns
        -------
        np.ndarray
            A 2D array with the aggregated values corresponding to the grid cells.

        Notes
        -----
        All subclasses currently accept the following two function arguments:
        - `fill_value`: The value used to fill grid cells containing no points.
        - `dtype`: The numpy data type for the aggregation result.

        Weighted aggregation strategies furthermore accept:
        - point_weights`: A 1D array of aggregation weights for all points.
        """
        pass

    def _assign_points_to_grid_cells(self, points):
        """
        Find the proper column and row indexes for all `points` and store
        them in `self.grid_col_ids` and `self.grid_row_ids`. Points outside
        the grid bounds are assigned a column and row index of -1.

        Parameters
        ----------
        points : np.ndarray
            2D array of point coordinates, where each row is (x, y).
        """
        xx = points[:, 0]
        yy = points[:, 1]

        # make a mask for points inside the grid bounds
        self.inside_mask = (xx >= self.layout.x_min) & (xx <= self.layout.x_max) & \
                           (yy >= self.layout.y_min) & (yy <= self.layout.y_max)

        num_oob = (~self.inside_mask).sum()  # type: ignore
        if num_oob and self.warn_out_of_bounds:
            warnings.warn(
                f"{num_oob} point(s) located outside the grid bounds. Set "
                f"`warn_out_of_bounds=False` to supress this warning.",
            )

        # initialise columns and row indexes as -1
        self.grid_col_ids = -1 * np.ones(xx.shape, dtype=int)
        self.grid_row_ids = -1 * np.ones(yy.shape, dtype=int)

        # use integer division to find points' colum and row index
        if np.any(self.inside_mask):
            inside_col_ids = (xx[self.inside_mask] - self.layout.x_min) // self.layout.cell_width
            inside_row_ids = (yy[self.inside_mask] - self.layout.y_min) // self.layout.cell_height

            # ensure that points with coordinates on the max boundaries
            # are assigned the correct column and row index
            inside_col_ids[xx[self.inside_mask] == self.layout.x_max] = -1
            inside_row_ids[yy[self.inside_mask] == self.layout.y_max] = -1

            self.grid_col_ids[self.inside_mask] = inside_col_ids
            self.grid_row_ids[self.inside_mask] = inside_row_ids

    def _aggregate_with_ufunc(
            self,
            np_ufunc=np.add,
            point_weights=None,
            init_value=0,
            dtype=np.float64
    ):
        """
        Perform a weighted aggregation of point weights within grid cells via
        `np.ufunc.at` for fast in-place aggregation.

        Parameters
        ----------
        np_ufunc : np.ufunc, optional
            A universal numpy function to be used in aggregation. Default is
            `np.add`. For a list of available functions, see
            https://numpy.org/doc/2.2/reference/ufuncs.html.
        point_weights : np.ndarray, optional
            A 1D array of weights for each point. Its length must match the
            number of points passed during class initialisation, (including
            out-of-bounds points). If no weights are provided (default), each
            point has weight 1 during aggregation.
        init_value : int or float, optional
            Value with which to initialise grid-cell aggregates before calling
            `np.ufunc.at`.
        dtype : np.dtype or obj, optional
            A valid numpy data type or object that can be converted to a numpy
            data type. Default is 'float'.  # TODO: Check

        Returns
        -------
        np.ndarray
            2D integer array
        """
        self._validate_point_weights(point_weights)

        # initialise aggregates with zeroes
        aggregates = np.full(self.layout.shape, fill_value=init_value, dtype=dtype)

        inside_col_ids = self.grid_col_ids[self.inside_mask]
        inside_row_ids = self.grid_row_ids[self.inside_mask]

        if point_weights is None:
            np_ufunc.at(aggregates, (inside_row_ids, inside_col_ids), 1)
        else:
            inside_weights = point_weights[self.inside_mask].astype(dtype)  # TODO: is this cast needed?
            np_ufunc.at(aggregates, (inside_row_ids, inside_col_ids), inside_weights)

        return aggregates

    def _simple_count(self):
        return self._aggregate_with_ufunc(np.add, point_weights=None, dtype=int)  # noqa

    def _weighted_sum(self, point_weights, dtype):
        return self._aggregate_with_ufunc(np.add, point_weights, dtype=dtype)  # noqa

    def _fill_results_for_empty_cells(self, aggregates, fill_value):
        counts = self._simple_count()
        empty = counts == 0
        aggregates[empty] = fill_value
        return aggregates

    def _validate_point_weights(self, point_weights):
        if point_weights is not None:
            if len(point_weights) != self.N:
                raise ValueError(
                    "Length mismatch. `point_weights` must provide exactly "
                    "one weight for each point passed upon class initialisation. "
                    f"(Expected array with shape ({self.N_inside},) but received "
                    f"{point_weights.shape}.)"
                )

    def plot(
            self,
            ax=None,
            colorbar=True,
            colorbar_kwargs=None,
            *,
            title=None,
            **kwargs
    ):
        """
        Use a heatmap to visualise gridded point-data aggregates.

        The heatmap is created via a call to `matplotlib.axes.Axes.imshow()`.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axis on which to plot the image. If None, a new figure and axis will
            be created.
        colorbar : bool, optional
            Whether to display a colorbar. Default is True.
        colorbar_kwargs : dict, optional
            Dictionary with keyword arguments passed to `matplotlib.pyplot.colorbar`
            for customising the appearance of the colorbar. None by default.
        title : str, optional
            Title for the plot. None by default.
        **kwargs : additional keyword arguments
            Additional keyword arguments passed to `matplotlib.axes.Axes.imshow` for
            customising the plot.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axis containing the heatmap plot.

        Raises
        ------
        ModuleNotFoundError
            If `matplotlib` is not installed.
        Warning
            If data aggregation has not yet been performed (i.e., `aggregate` has not
            yet been called).

        Notes
        -----
        - The extent of the plot is automatically set to match the grid’s spatial bounds,
          but this can be customised by passing the `extent` argument in `kwargs`.
        - The color map (`cmap`) defaults to 'bone_r', but this can be overwritten.
        """
        if self.cell_aggregates is None:
            warnings.warn("Call `aggregate()` before plotting.")
            return

        # set imshow defaults
        domain_extent = [self.layout.x_min, self.layout.x_max, self.layout.y_min, self.layout.y_max]
        kwargs['cmap'] = kwargs.get('cmap', 'bone_r')
        kwargs['origin'] = kwargs.get('origin', 'lower')
        kwargs['extent'] = kwargs.get('extent', domain_extent)

        # create new axis unless user passes one
        if ax is None:
            fig, ax = plt.subplots()

        # plot
        im = ax.imshow(self.cell_aggregates, **kwargs)
        if title is not None:
            ax.set_title(title)

        if colorbar:
            ckwargs = {} if colorbar_kwargs is None else colorbar_kwargs
            plt.colorbar(im, ax=ax, **ckwargs)
        else:
            if colorbar_kwargs is not None:
                warnings.warn('colorbar_kwargs are ignored when colorbar=False.')

        return ax


class CountAggregator(BasePointAggregator):
    """Simply counts the number of points within grid cells."""

    def aggregate(self):
        return self._simple_count()


class WeightedSumAggregator(BasePointAggregator):
    """
    Computes a weighted sum over points within grid cells, based on a
    user-provided array of aggregation weights for all points. Cells
    without any points are represented with a customisable fill value
    (default np.nan).
    """

    def aggregate(
            self,
            point_weights,
            fill_value=np.nan,
            dtype=np.float64,
    ):
        weighted_sums = self._weighted_sum(point_weights, dtype=dtype)
        return self._fill_results_for_empty_cells(weighted_sums, fill_value)


class WeightedAverageAggregator(BasePointAggregator):
    """
    Computes a weighted average over points within grid cells, based on
    a user-provided array of aggregation weights for all points. Cells
    without any points are represented with a customisable fill value
    (default np.nan).
    """

    def aggregate(
            self,
            point_weights,
            fill_value=np.nan,
            dtype=np.float64
    ):
        counts = self._simple_count()
        aggregates = self._weighted_sum(point_weights, dtype=dtype)

        # compute averages for cells with events
        with_events = counts > 0
        aggregates[with_events] = aggregates[with_events] / counts[with_events]

        # set fill value for cells without events
        aggregates[~with_events] = fill_value

        return aggregates


class MinimumWeightAggregator(BasePointAggregator):
    """
    Finds the minimum weight of all points within grid cells, based on
    a user-provided array of aggregation weights for all points. Cells
    without any points are represented with a customisable fill value
    (default np.nan).
    """

    def aggregate(
            self,
            point_weights,
            fill_value=np.nan,
            dtype=np.float64,
    ):
        minima = self._aggregate_with_ufunc(np.minimum, point_weights, init_value=np.inf)  # noqa
        return self._fill_results_for_empty_cells(minima, fill_value)


class MaximumWeightAggregator(BasePointAggregator):
    """
    Finds the maximum weight of all points within grid cells, based on
    a user-provided array of aggregation weights for all points. Cells
    without any points are represented with customisable a fill value
    (default np.nan).
    """

    def aggregate(
            self,
            point_weights,
            fill_value=np.nan,
            dtype=np.float64,
    ):
        maxima = self._aggregate_with_ufunc(np.maximum, point_weights, init_value=-np.inf)  # noqa
        return self._fill_results_for_empty_cells(maxima, fill_value)


class PointLocaliser(BasePointAggregator):
    """
    Locates points on the grid, but performs no data aggregation.
    The `cell_aggregates` attribute will hence be `None`.
    """
    _check_implementation = False

    def aggregate(self):
        return None
