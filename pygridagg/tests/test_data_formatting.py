import numpy as np
import pytest

from pygridagg.aggregate import ensure_array_shape


def test_unknown_dtype_raises():
    data_np = np.random.randn(5, 2)
    with pytest.raises(ValueError):
        ensure_array_shape(data_np.tolist())  # type: ignore


def test_unknown_array_shape_raises():
    weired_np = np.random.randn(5)
    with pytest.raises(ValueError):
        ensure_array_shape(weired_np)
