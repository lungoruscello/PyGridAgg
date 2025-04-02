"""
Fast aggregation of 2D points on spatial grids.
========================================================================
"""
import warnings

import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    PLOTTING_AVAILABLE = True
except ModuleNotFoundError:
    PLOTTING_AVAILABLE = False

# TODO: Add from_bbox method


class FlexibleGridLayout:
    """
    A 2D grid layout in which the number of columns and rows, as well as
    the grid's total width and height can be set independently of each other.

    This class defines a grid structure, computes grid-cell centroids, but
    does not hold any point data.


    Attributes
    ----------
    x_centroids : np.ndarray
        Sorted 1D array with x coordinates of grid-cell centroids
    y_centroids : np.ndarray
        Sorted 1D array with y coordinates of grid-cell centroids
    """

    def __init__(
            self,
            total_width,
            total_height,
            grid_center,
            num_cols,
            num_rows
    ) -> None:
        """
        Initialise a `SpatialGrid` with the given parameters.

        Parameters
        ----------
        total_width : int or float
            The total width of the grid.
        total_height : int or float
            The total height of the grid.
        grid_center : Tuple
            A tuple specifying the x and y coordinates of the grid's center point.
        num_cols : int
            The number of columns in the grid.
        num_rows : int
            The number of rows in the grid.

        Raises
        -------
        ValueError
            If one of `num_cols`, `num_rows`, `total_width`, or `total_height`
            is not strictly positive.
        """
        if not isinstance(grid_center, tuple):
            raise ValueError(
                "`grid_center` must be a tuple specifying the x and y "
                "coordinate of the grid's desired centre point"
            )

        if num_cols <= 0 or num_rows <= 0:
            raise ValueError("Number of columns and rows must be positive integers.")

        if total_width <= 0 or total_height <= 0:
            raise ValueError("Grid width and height must be positive.")

        self.num_cols = num_cols
        self.num_rows = num_rows
        self.total_width = total_width
        self.total_height = total_height
        self.num_cells = self.num_cols * self.num_rows
        self.shape = (self.num_rows, self.num_cols)

        # compute grid bounds
        c_x, c_y = grid_center
        half_width = self.total_width / 2
        half_height = self.total_height / 2
        self.x_min, self.x_max = c_x - half_width, c_x + half_width
        self.y_min, self.y_max = c_y - half_height, c_y + half_height

        # compute grid-cell attributes
        self.cell_width = (self.x_max - self.x_min) / num_cols
        self.cell_height = (self.y_max - self.y_min) / num_rows
        self.cell_area = self.cell_width * self.cell_height

        # compute centroids of all grid cells
        xx = np.linspace(self.x_min, self.x_max - self.cell_width, num_cols)
        yy = np.linspace(self.y_min, self.y_max - self.cell_height, num_rows)
        self.x_centroids = xx + (self.cell_width / 2)
        self.y_centroids = yy + (self.cell_height / 2)


class SquareGridLayout(FlexibleGridLayout):
    """
    A square spatial grid with an equal number of columns and rows.

    This class defines a grid structure, computes grid-cell centroids, but
    does not hold any point data.
    """

    @classmethod
    def from_points(cls, points, padding_percentage=.001, **kwargs):
        """
        Create a `SquareGridLayout` that covers the spatial extent of the provided `points`.

        Parameters
        ----------
        points : np.ndarray
            Array with shape (N, 2) holding x and y coordinates for a collection of N points.
        padding_percentage : float, optional
            Amount of padding to add to the true spatial extent of the provided `points`.
            Adding a small amount of padding helps prevent out-of-bounds points due to
            floating point imprecision. Default is 0.001 (i.e., 0.1%).

        **kwargs
            Additional arguments used to initialise the `SquareGridLayout`.

        Returns
        -------
        SquareGridLayout
            A class instance created from the bounding box of a point collection.
        """
        dstats = infer_spatial_domain_stats(points)
        centre = (dstats['center_x'], dstats['center_y'])
        L = (1 + padding_percentage) * dstats['max_extent']

        return cls(
            total_side_length=L,
            grid_center=centre,
            **kwargs)

    def __init__(
            self,
            total_side_length,
            grid_center,
            num_cells,
    ):
        """
        Initialise a `SquareGridLayout` with the given parameters.

        Parameters
        ----------
        total_side_length : int or float
            The total side length (=width and height) of the grid.
        grid_center : Tuple
            A tuple specifying the x and y coordinates of the grid's center point.
        num_cells : int
            The total number of grid cells. The square-root of `num_cells` must be
            an integer.

        Raises
        -------
        ValueError
            If `num_cells` is not a perfect square.
        """

        num_cols = np.sqrt(num_cells)
        if num_cols % 1 != 0:
            raise ValueError(
                f"Invalid `num_cells={num_cells}`: Must be a perfect "
                f"square (e.g., 9, 16, 25, 100)."
            )

        super().__init__(
            total_width=total_side_length,
            total_height=total_side_length,
            grid_center=grid_center,
            num_cols=int(num_cols),
            num_rows=int(num_cols)
        )


class PointAggregator:
    """
    Aggregates 2D points on a spatial grid. The aggregation method can either
    be simple counting or a weighted sum.

    Attributes
    ----------
    grid_col_ids : np.ndarray
        1D Array of grid-column indexes for each point. A column index of -1
        marks out-of-bounds points.
    grid_row_ids : np.ndarray
        1D Array of grid-row indexes for each point. A row index of -1 marks
        out-of-bounds points.
    inside_mask : np.ndarray
        1D Boolean mask indicating points located within the grid bounds.
    cell_aggregates : np.ndarray or None
        2D Gridded point-data aggregates, or None if no aggregation was performed.
        (Read-only property).

    Methods
    -------
    __init__(grid, points, localise_only=False, warn_out_of_bounds=True)
        Initialises the `PointAggregator` with a grid layout and a collection of points.
    aggregate(point_weights=None)
        Aggregates point data across grid cells, either by counting or summing weights.
    plot(ax=None, colorbar=True, colorbar_kwargs=None, **kwargs)
        Visualises point-data aggregates with a heatmap.
    """

    def __init__(
            self,
            grid_layout,
            points,
            *,
            point_weights=None,
            localise_only=False,
            warn_out_of_bounds=True
    ):
        """
        Initialise a `PointAggregator` with the given grid and points.

        Parameters
        ----------
        grid_layout : FlexibleGridLayout
            The spatial grid on which points will be aggregated.
        points : np.ndarray
            Array with shape (N, 2) holding x and y coordinates for a collection of N points.
        point_weights : np.ndarray or None, optional
            An optional 1D array of aggregation weights for each point. If no weights are
            provided (default), point-data is aggregated using simple counting. Note that
            aggregation weights can also be negative.
        localise_only : bool, optional
            Whether to skip point-data aggregation during class initialisation.
            Default is False. If True, the provided `points` will still be 'localised'
            on the grid and users will be able to retrieve their column and row indexes.
            However, the `cell_aggregates` property will be None.
        warn_out_of_bounds : bool, optional
            Whether to show a warning for points that are out-of-bounds. Default
            is True. If False, no such warning will be shown.
        """
        self.grid = grid_layout
        self.warn_out_of_bounds = warn_out_of_bounds

        points = _ensure_array_shape(points)
        self._assign_points_to_grid_cells(points)

        # initialise point-data aggregates to None
        self._aggregates = None  # will be overwritten by `aggregate`

        if not localise_only:
            self.aggregate(point_weights)
        else:
            if point_weights is not None:
                warnings.warn(
                    "`point_weights` are ignored when `localise_only` is True."
                )

    @property
    def cell_aggregates(self):
        """
        Read-only property returning the gridded point-data aggregates
        (either simple counts or weighted sums), if available.

        Returns
        -------
        np.ndarray or None
            The point-data aggregates, or None if no aggregation has yet been
            performed.
        """
        return self._aggregates

    def _assign_points_to_grid_cells(self, points):
        """
        Find the proper column and row indexes for all `points`. Points
        outside the grid bounds receive a column and row index of -1.

        Parameters
        ----------
        points : np.ndarray
            2D array of point coordinates, where each row is (x, y).
        """
        self._no_points = points.shape[0] == 0

        if self._no_points:
            # nothing to do
            self.grid_col_ids = np.array([])
            self.grid_row_ids = np.array([])
            self.inside_mask = np.array([])
            return None

        xx = points[:, 0]
        yy = points[:, 1]

        # make a mask for points inside the grid bounds
        in_x = (xx >= self.grid.x_min) & (xx <= self.grid.x_max)
        in_y = (yy >= self.grid.y_min) & (yy <= self.grid.y_max)
        self.inside_mask = in_x & in_y

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
            inside_col_ids = (xx[self.inside_mask] - self.grid.x_min) // self.grid.cell_width
            inside_row_ids = (yy[self.inside_mask] - self.grid.y_min) // self.grid.cell_height

            # ensure that points with coordinates on the max boundaries
            # are assigned the correct column and row index
            inside_col_ids[xx[self.inside_mask] == self.grid.x_max] = -1
            inside_row_ids[yy[self.inside_mask] == self.grid.y_max] = -1

            self.grid_col_ids[self.inside_mask] = inside_col_ids
            self.grid_row_ids[self.inside_mask] = inside_row_ids


    def aggregate(self, point_weights=None):
        """
        Aggregate point data within grid cells using either a simple count or a weighted sum.

        Parameters
        ----------
        point_weights : np.ndarray or None, optional
            1D array of weights to be used during point aggregation. If no weights are
            provided (default), point data is aggregated by simple counting. Otherwise,
            aggregation results will be a weighted sum. If provided, the length of
            `point_weights` must match the overall number of points passed during
            class initialisation (incl. points that are out of bounds, if any).

        Returns
        -------
        np.ndarray
            A 2D array of gridded point-data aggregates.

        Raises
        ------
        BadDataError
            If the length of `point_weights` does not match the number of points.
        """

        N_inside = self.inside_mask.sum()  # total points inside spatial domain

        if point_weights is not None:
            if len(point_weights) != N_inside:
                raise ValueError(
                    "Length mismatch. `point_weights` must provide exactly "
                    "one weight for each point passed upon class initialisation. "
                    f"(Expected array with shape ({N_inside},) but received "
                    f"{point_weights.shape}.)"
                )

        # initialise raster with counts of zero
        self._aggregates = np.zeros(self.grid.shape, dtype=int)

        if not self._no_points:
            inside_row_ids = self.grid_row_ids[self.inside_mask]
            inside_col_ids = self.grid_col_ids[self.inside_mask]

            if point_weights is None:
                np.add.at(self._aggregates, (inside_row_ids, inside_col_ids), 1)  # noqa
                assert self._aggregates.sum() == N_inside  # totals must match
            else:
                inside_weights = point_weights[self.inside_mask]
                weighted_counts = np.zeros(self.grid.shape, dtype=float)
                np.add.at(weighted_counts, (inside_row_ids, inside_col_ids), inside_weights)  # noqa
                assert weighted_counts.sum() == inside_weights.sum()  # weighted sums must match
                self._aggregates = weighted_counts

        return self._aggregates

    def plot(self, ax=None, colorbar=True, colorbar_kwargs=None, **kwargs):
        """
        Use a heatmap to visualise gridded point-data aggregates.

        The heatmap is created via a call to `matplotlib.axes.Axes.imshow()`.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axis on which to plot the image. If None, a new figure and axis will
            be created.
        colorbar : bool, optional, default=True
            Whether to display a colorbar.
        colorbar_kwargs : dict, optional, default=None
            Additional keyword arguments passed to `matplotlib.pyplot.colorbar` for
            customising the appearance of the colorbar.
        **kwargs : additional keyword arguments
            Additional keyword arguments passed to `matplotlib.pyplot.colorbar` for
            customising the colorbar.

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
        - The grid extent is automatically determined from the grid’s spatial bounds, but
          this can be customised using the `extent` argument in `kwargs`.
        - The color map (`cmap`) defaults to 'bone_r', but this can be overwritten.
        """
        if self.cell_aggregates is None:
            warnings.warn('Nothing to plot. Call `aggregate()` first.')
            return None

        if not PLOTTING_AVAILABLE:
            raise ModuleNotFoundError(
                "The optional 'matplotlib' library is required for plotting. "
                "Please install it using pip, conda, or mamba."
            )

        # set imshow defaults
        domain_extent = [self.grid.x_min, self.grid.x_max, self.grid.y_min, self.grid.y_max]
        kwargs['cmap'] = kwargs.get('cmap', 'bone_r')
        kwargs['origin'] = kwargs.get('origin', 'lower')
        kwargs['extent'] = kwargs.get('extent', domain_extent)

        # create new axis unless user passes one
        if ax is None:
            fig, ax = plt.subplots()

        # plot
        im = ax.imshow(self.cell_aggregates, **kwargs)

        if colorbar:
            ckwargs = {} if colorbar_kwargs is None else colorbar_kwargs
            plt.colorbar(im, ax=ax, **ckwargs)
        else:
            if colorbar_kwargs is not None:
                warnings.warn('colorbar_kwargs are ignored when colorbar=False.')

        return ax


def infer_spatial_domain_stats(points):
    """
    Infer spatial domain statistics from a collection of points.

    Parameters
    ----------
    points : np.ndarray
        Array with shape (N, 2) holding x and y coordinates for a collection of N points.

    Returns
    -------
    dict of str to float
        A dictionary holding information on the spatial extent and centre point
        of the bounding box that contains all `points`.

    Raises
    ------
    BadDataError
        If `points` is empty.
    """
    if points.size == 0:
        raise ValueError('Cannot infer spatial domain from empty point data.')
    points = _ensure_array_shape(points)

    min_x, min_y = points.min(axis=0)  # type: ignore
    max_x, max_y = points.max(axis=0)  # type: ignore
    ext_x = max_x - min_x
    ext_y = max_y - min_y

    return dict(
        x_extent=ext_x,
        y_extent=ext_y,
        center_x=min_x + (ext_x / 2),
        center_y=min_y + (ext_y / 2),
        max_extent=max(ext_x, ext_y)
    )


def _ensure_array_shape(points):
    if not isinstance(points, np.ndarray):
        raise ValueError(
            "Unsupported data type. Point coordinates must be provided as a "
            f"numpy array, but you passed an instance of type {type(points)}."
        )

    if len(points.shape) != 2:
        raise ValueError(
            f"Wrong array shape ({points.shape}). Point coordinates must be provided "
            "as an array with shape (N, D), where D >= 2. The first dimension (N) "
            "represents the number of points, and the second dimension (D) should "
            "contain at least two values: points' x and y coordinates."
        )

    if points.shape[1] > 2:
        N, D = points.shape
        warnings.warn(
            f"Warning: Point data has shape (N={N}, D={D}). Only the first two "
            f"d-dimensions (x and y coordinates) will be used. Data in additional "
            f"dimensions will be ignored."
        )

    return points[:, :2]
