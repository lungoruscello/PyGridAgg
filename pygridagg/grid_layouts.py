"""
Customisable grid layouts that can be used in subsequent data aggregation.
========================================================================
"""
import numpy as np

from utils import infer_spatial_domain_stats

__all__ = [
    "FlexibleGridLayout",
    "SquareGridLayout"
]


# TODO: Add from_bbox method
# TODO: Make convenience constructors available to FlexibleGridLayout

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
            The total number of grid cells, which must be a perfect square (e.g., 9, 25, 36,...)

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
