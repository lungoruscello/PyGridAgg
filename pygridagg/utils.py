import numpy as np
import warnings


def infer_spatial_domain_stats(points):
    """
    Infer spatial domain statistics from a collection of points.

    Parameters
    ----------
    points : np.ndarray
        Array with shape (N, 2) holding x and y coordinates for a collection
        of N points.

    Returns
    -------
    dict of str to float
        A dictionary holding information on the spatial extent and centre point
        of the bounding box that contains all `points`.

    Raises
    ------
    ValueError
        If `points` is empty.
    """
    if points.size == 0:
        raise ValueError('Cannot infer spatial domain from empty point data.')
    points = ensure_array_shape(points)

    min_x, min_y = points.min(axis=0)  # type: ignore
    max_x, max_y = points.max(axis=0)  # type: ignore
    ext_x = max_x - min_x
    ext_y = max_y - min_y

    return dict(
        x_min=min_x,
        y_min=min_y,
        x_max=max_x,
        y_max=max_y,
        x_extent=ext_x,
        y_extent=ext_y,
        x_center=min_x + (ext_x / 2),
        y_center=min_y + (ext_y / 2),
        max_extent=max(ext_x, ext_y)
    )


def ensure_array_shape(points):
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

    if points.shape[0] == 0:
        raise ValueError(f"Point-data array is empty.")

    if points.shape[1] > 2:
        N, D = points.shape
        warnings.warn(
            f"Warning: Point data has shape (N={N}, D={D}). Only the first two "
            f"d-dimensions (x and y coordinates) will be used. Data in additional "
            f"dimensions will be ignored."
        )

    return points[:, :2]
