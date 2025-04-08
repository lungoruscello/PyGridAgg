"""
Customisable grid layouts that can be used in subsequent data aggregation.
========================================================================
"""
import numpy as np
import warnings
from .utils import infer_spatial_domain_stats

__all__ = [
    "FlexibleGridLayout",
    "SquareGridLayout"
]


class FlexibleGridLayout:
    """
    A 2D grid layout where the number of columns and rows, as well as
    the total width and height of the grid, can be set independently.

    This class defines a grid structure, computes grid-cell centroids, but
    does not hold any point data.

    Attributes
    ----------
    x_centroids : np.ndarray
        Sorted 1D array with x coordinates of grid-cell centroids
    y_centroids : np.ndarray
        Sorted 1D array with y coordinates of grid-cell centroids
    x_min : float
        Minimum x-coordinate of the grid boundary.
    x_max : float
        Maximum x-coordinate of the grid boundary.
    y_min : float
        Minimum y-coordinate of the grid boundary.
    y_max : float
        Maximum y-coordinate of the grid boundary.
    shape : tuple of (int, int)
        A tuple representing the size of the grid in terms of (rows, columns).
    """

    def __init__(self, x_min, x_max, y_min, y_max, num_cols=10, num_rows=10):
        """
        Create a `FlexibleGridLayout` from a bounding box.

        Parameters
        ----------
        x_min : float
            Minimum x-coordinate of the bounding box.
        x_max : float
            Maximum x-coordinate of the bounding box.
        y_min : float
            Minimum y-coordinate of the bounding box.
        y_max : float
            Maximum y-coordinate of the bounding box.
        num_cols : int
            The number of columns in the grid. Default is 10.
        num_rows : int
            The number of rows in the grid. Default is 10.

        Raises
        -------
        ValueError
            If one of `num_cols` or `num_rows` is not strictly positive.
            If x_max is not greater than x_min.
            If y_max is not greater than y_min.
        """
        if num_cols <= 0 or num_rows <= 0:
            raise ValueError("Number of columns and rows must be positive integers.")

        if x_min >= x_max:
            raise ValueError("'x_min' must be smaller than 'x_max'.")

        if y_min >= y_max:
            raise ValueError("'y_min' must be smaller than 'y_max'.")

        self.num_cols = num_cols
        self.num_rows = num_rows
        self.total_width = x_max - x_min
        self.total_height = y_max - y_min
        self.num_cells = self.num_cols * self.num_rows
        self.shape = (self.num_rows, self.num_cols)
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

        # compute grid-cell attributes
        self.cell_width = (self.x_max - self.x_min) / num_cols
        self.cell_height = (self.y_max - self.y_min) / num_rows
        self.cell_area = self.cell_width * self.cell_height  # not used

        # compute centroids of all grid cells
        xx = np.linspace(self.x_min, self.x_max - self.cell_width, num_cols)
        yy = np.linspace(self.y_min, self.y_max - self.cell_height, num_rows)
        self.x_centroids = xx + (self.cell_width / 2)
        self.y_centroids = yy + (self.cell_height / 2)

    @classmethod
    def from_centroid(cls, total_width, total_height, grid_center, num_cols=10, num_rows=10):
        """
        Create a `FlexibleGridLayout` from a centre coordinate and side dimensions.

        Parameters
        ----------
        total_width : int or float
            The total width of the grid.
        total_height : int or float
            The total height of the grid.
        grid_center : Tuple[float, float]
            A tuple specifying the x and y coordinates of the grid's center point.
        num_cols : int
            The number of columns in the grid. Default is 10.
        num_rows : int
            The number of rows in the grid. Default is 10.

        Returns
        -------
        FlexibleGridLayout

        Raises
        -------
        ValueError
            If `grid_center` is not a tuple.
        """
        if not isinstance(grid_center, tuple):
            raise ValueError(
                "`grid_center` must be a tuple specifying the x and y "
                "coordinate of the grid's desired centre point"
            )


        c_x, c_y = grid_center
        x_min = c_x - (total_width / 2.)
        x_max = c_x + (total_width / 2.)
        y_min = c_y - (total_height / 2.)
        y_max = c_y + (total_height / 2.)

        # create and return the `FlexibleGridLayout`
        return cls(x_min, x_max, y_min, y_max, num_cols, num_rows)

    @classmethod
    def from_unit_square(cls, num_cols=10, num_rows=10):  # noqa
        """
        Create a `FlexibleGridLayout` covering the unit square.

        Parameters
        ----------
        num_cols : int
            The number of columns in the grid. Default is 10.
        num_rows : int
            The number of rows in the grid. Default is 10.

        Returns
        -------
        FlexibleGridLayout
        """
        return cls(0, 1, 0, 1, num_cols, num_rows)

    @classmethod
    def from_points(cls, points, num_cols=10, num_rows=10, padding_percent=0.0):
        """
        Create a `FlexibleGridLayout` that encompasses all provided `points`.

        Parameters
        ----------
        points : np.ndarray
            Array with shape (N, 2) holding x and y coordinates for a collection of N points.
        num_cols : int, optional
            The number of columns in the grid. Default is 10.
        num_rows : int, optional
            The number of rows in the grid. Default is 10.
        padding_percent : float, optional
            Amount of padding to add to the true spatial extent of the provided `points`,
            expressed in percent (i.e., 10 means 10%).

        Returns
        -------
        FlexibleGridLayout
        """

        dstats = infer_spatial_domain_stats(points)
        if padding_percent == 0:
            # bbox is already given
            bbox = (dstats['x_min'], dstats['x_max'], dstats['y_min'], dstats['y_max'])
        else:
            # make padded bbox
            padding = 1 + (padding_percent / 100.0)
            W = padding * dstats['x_extent']
            H = padding * dstats['y_extent']
            x_min = dstats['x_center'] - (W / 2.)
            x_max = dstats['x_center'] + (W / 2.)
            y_min = dstats['y_center'] - (H / 2.)
            y_max = dstats['y_center'] + (H / 2.)
            bbox = (x_min, x_max, y_min, y_max)

        # create and return the `FlexibleGridLayout`
        return cls(*bbox, num_cols, num_rows)

    def summary(self):
        """Print summary information about the grid layout."""
        print(f"Summary of `{self.__class__.__name__}`:")
        print(f"  Dimensions: {self.total_width} x {self.total_height}")
        print(f"  Shape (rows, columns): {self.shape}")
        print(f"  No. of grid cells: {self.num_cells}")
        print(f"  Grid bounds:")
        print(f"    X Min: {self.x_min:.2f}, X Max: {self.x_max:.2f}")
        print(f"    Y Min: {self.y_min:.2f}, Y Max: {self.y_max:.2f}")


class SquareGridLayout(FlexibleGridLayout):
    """
    A square spatial grid with an equal number of columns and rows.

    This class defines a grid structure, computes grid-cell centroids, but
    does not hold any point data.

    Attributes
    ----------
    x_centroids : np.ndarray
        Sorted 1D array with x coordinates of grid-cell centroids
    y_centroids : np.ndarray
        Sorted 1D array with y coordinates of grid-cell centroids
    x_min : float
        Minimum x-coordinate of the grid boundary.
    x_max : float
        Maximum x-coordinate of the grid boundary.
    y_min : float
        Minimum y-coordinate of the grid boundary.
    y_max : float
        Maximum y-coordinate of the grid boundary.
    shape : tuple of (int, int)
        A tuple representing the size of the grid in terms of (rows, columns).
    """

    def __init__(self, x_min, x_max, y_min, y_max, num_cells=64, warn=True):
        """
        Create a `SquareGridLayout` from a bounding box that is converted
        into a square by expanding the shorter dimension (if any).

        Parameters
        ----------
        x_min : float
            Minimum x-coordinate of the bounding box.
        x_max : float
            Maximum x-coordinate of the bounding box.
        y_min : float
            Minimum y-coordinate of the bounding box.
        y_max : float
            Maximum y-coordinate of the bounding box.
        num_cells : int
            The total number of grid cells, which must be a perfect square (e.g., 9,
            25, 36,...). Default is 64.
        warn : bool, optional
            Whether to issue a warning when the bounding box is not square and needs
            to be enlarged along its shorter dimension. Default is True.

        Raises
        -------
        ValueError
            If `num_cells` is not a perfect square.
        """
        x_ext = x_max - x_min
        y_ext = y_max - y_min

        if not np.isclose(x_ext, y_ext):
            if warn:
                warnings.warn(
                    "I am converting the specified bounding box into a square. "
                    "If you do not want a square grid, with equal height and "
                    "width, please use the `FlexibleGridLayout` instead."
                )
            if y_ext < x_ext:
                c_y = y_max - (y_ext / 2)
                y_min = c_y - (x_ext / 2)
                y_max = c_y + (x_ext / 2)
                assert np.isclose(y_max - y_min, x_ext)
            else:
                c_x = x_max - (x_ext / 2)
                x_min = c_x - (y_ext / 2)
                x_max = c_x + (y_ext / 2)
                assert np.isclose(x_max - x_min, y_ext)

        num_cols = np.sqrt(num_cells)
        if num_cols % 1 != 0:
            raise ValueError(
                f"Invalid `num_cells={num_cells}`: Must be a perfect "
                f"square (e.g., 9, 16, 25, 100)."
            )
        super().__init__(
            x_min, x_max, y_min, y_max,
            num_cols=int(num_cols), num_rows=int(num_cols)
        )

    @classmethod
    def from_centroid(cls, total_side_length, grid_center, num_cells=64):  # noqa
        """
        Create a `SquareGridLayout` from a centre coordinate and side dimensions.

        Parameters
        ----------
        total_side_length : int or float
            The total side length (=width and height) of the grid.
        grid_center : Tuple[float, float]
            A tuple specifying the x and y coordinates of the grid's center point.
        num_cells : int
            The total number of grid cells, which must be a perfect square (e.g., 9,
            25, 36,...). Default is 64.

        Returns
        -------
        SquareGridLayout

        Raises
        -------
        ValueError
            If `grid_center` is not a tuple.
        """
        if not isinstance(grid_center, tuple):
            raise ValueError(
                "`grid_center` must be a tuple specifying the x and y "
                "coordinate of the grid's desired centre point"
            )

        c_x, c_y = grid_center
        half_length = total_side_length / 2
        x_min = c_x - half_length
        x_max = c_x + half_length
        y_min = c_y - half_length
        y_max = c_y + half_length

        # create and return the `SquareGridLayout`
        return cls(x_min, x_max, y_min, y_max, num_cells, warn=False)

    @classmethod
    def from_unit_square(cls, num_cells=64):  # noqa
        """
        Create a `SquareGridLayout` covering the unit square.

        Parameters
        ----------
        num_cells : int
            The total number of grid cells, which must be a perfect square (e.g., 9,
            25, 36,...). Default is 64.

        Returns
        -------
        SquareGridLayout
        """
        return cls(0, 1, 0, 1, num_cells, warn=False)

    @classmethod
    def from_points(cls, points, num_cells=64, padding_percent=.001):  # noqa
        """
        Create a `SquareGridLayout` from the bounding box of the provided
        `points` while expanding the boxes' shorter dimension (if any)
        to create a square.

        Parameters
        ----------
        points : np.ndarray
            Array with shape (N, 2) holding x and y coordinates for a collection of N points.
        num_cells : int
            The total number of grid cells, which must be a perfect square (e.g., 9,
            25, 36,...). Default is 64.
        padding_percent : float, optional
            Amount of padding to add to the true spatial extent of the provided `points`.
            Adding a small amount of padding helps prevent out-of-bounds points due to
            floating point imprecision. Default is 0.001 (i.e., 0.1%).

        Returns
        -------
        SquareGridLayout
            A class instance created from the bounding box of a point collection.
        """
        dstats = infer_spatial_domain_stats(points)
        if padding_percent == 0:
            # bbox is already given
            bbox = (dstats['x_min'], dstats['x_max'], dstats['y_min'], dstats['y_max'])
        else:
            # make padded bbox
            padding = 1 + (padding_percent / 100.0)
            L = padding * dstats['max_extent']
            half_length = L / 2
            x_min = dstats['x_center'] - half_length
            x_max = dstats['x_center'] + half_length
            y_min = dstats['y_center'] - half_length
            y_max = dstats['y_center'] + half_length
            bbox = (x_min, x_max, y_min, y_max)

        # create and return the `SquareGridLayout`
        return cls(*bbox, num_cells, warn=False)
