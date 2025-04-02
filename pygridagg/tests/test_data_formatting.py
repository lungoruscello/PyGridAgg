import numpy as np
import pytest

from pygridagg.aggregate import _validate_point_array


def test_unknown_dtype_raises():
    data_np = np.random.randn(5, 2)
    with pytest.raises(ValueError):
        _validate_point_array(data_np.tolist())  # type: ignore


def test_unknown_array_shape_raises():
    weired_np = np.random.randn(5)
    with pytest.raises(ValueError):
        _validate_point_array(weired_np)
