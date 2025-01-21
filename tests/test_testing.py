from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

from array_api_extra._lib import Backend
from array_api_extra._lib._testing import xp_assert_close, xp_assert_equal

# mypy: disable-error-code=no-any-decorated
# pyright: reportUnknownParameterType=false,reportMissingParameterType=false

param_assert_equal_close = pytest.mark.parametrize(
    "func",
    [
        xp_assert_equal,
        pytest.param(
            xp_assert_close,
            marks=pytest.mark.skip_xp_backend(Backend.SPARSE, reason="no isdtype"),
        ),
    ],
)


@param_assert_equal_close
def test_assert_close_equal_basic(xp: ModuleType, func: Callable[..., None]):  # type: ignore[no-any-explicit]
    func(xp.asarray(0), xp.asarray(0))
    func(xp.asarray([1, 2]), xp.asarray([1, 2]))

    with pytest.raises(AssertionError, match="shapes do not match"):
        func(xp.asarray([0]), xp.asarray([[0]]))

    with pytest.raises(AssertionError, match="dtypes do not match"):
        func(xp.asarray(0, dtype=xp.float32), xp.asarray(0, dtype=xp.float64))

    with pytest.raises(AssertionError):
        func(xp.asarray([1, 2]), xp.asarray([1, 3]))

    with pytest.raises(AssertionError, match="hello"):
        func(xp.asarray([1, 2]), xp.asarray([1, 3]), err_msg="hello")


@pytest.mark.skip_xp_backend(Backend.NUMPY, reason="test other ns vs. numpy")
@pytest.mark.skip_xp_backend(Backend.NUMPY_READONLY, reason="test other ns vs. numpy")
@param_assert_equal_close
def test_assert_close_equal_namespace(xp: ModuleType, func: Callable[..., None]):  # type: ignore[no-any-explicit]
    with pytest.raises(AssertionError):
        func(xp.asarray(0), np.asarray(0))
    with pytest.raises(TypeError):
        func(xp.asarray(0), 0)
    with pytest.raises(TypeError):
        func(xp.asarray([0]), [0])


@pytest.mark.skip_xp_backend(Backend.SPARSE, reason="no isdtype")
def test_assert_close_tolerance(xp: ModuleType):
    xp_assert_close(xp.asarray([100.0]), xp.asarray([102.0]), rtol=0.03)
    with pytest.raises(AssertionError):
        xp_assert_close(xp.asarray([100.0]), xp.asarray([102.0]), rtol=0.01)

    xp_assert_close(xp.asarray([100.0]), xp.asarray([102.0]), atol=3)
    with pytest.raises(AssertionError):
        xp_assert_close(xp.asarray([100.0]), xp.asarray([102.0]), atol=1)


@param_assert_equal_close
@pytest.mark.skip_xp_backend(Backend.SPARSE, reason="no bool indexing by sparse arrays")
def test_assert_close_equal_none_shape(xp: ModuleType, func: Callable[..., None]):  # type: ignore[no-any-explicit]
    """On dask and other lazy backends, test that a shape with NaN's or None's
    can be compared to a real shape.
    """
    a = xp.asarray([1, 2])
    a = a[a > 1]

    func(a, xp.asarray([2]))
    with pytest.raises(AssertionError):
        func(a, xp.asarray([2, 3]))
    with pytest.raises(AssertionError):
        func(a, xp.asarray(2))
    with pytest.raises(AssertionError):
        func(a, xp.asarray([3]))

    # Swap actual and desired
    func(xp.asarray([2]), a)
    with pytest.raises(AssertionError):
        func(xp.asarray([2, 3]), a)
    with pytest.raises(AssertionError):
        func(xp.asarray(2), a)
    with pytest.raises(AssertionError):
        func(xp.asarray([3]), a)
