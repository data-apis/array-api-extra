from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

from array_api_extra._lib import Backend
from array_api_extra._lib._testing import xp_assert_close, xp_assert_equal
from array_api_extra._lib._utils._compat import (
    array_namespace,
    is_dask_namespace,
    is_jax_namespace,
)
from array_api_extra._lib._utils._typing import Array
from array_api_extra.testing import lazy_xp_function

# mypy: disable-error-code=no-any-decorated
# pyright: reportUnknownParameterType=false,reportMissingParameterType=false

param_assert_equal_close = pytest.mark.parametrize(
    "func",
    [
        xp_assert_equal,
        pytest.param(
            xp_assert_close,
            marks=pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no isdtype"),
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
@pytest.mark.parametrize("func", [xp_assert_equal, xp_assert_close])
def test_assert_close_equal_namespace(xp: ModuleType, func: Callable[..., None]):  # type: ignore[no-any-explicit]
    with pytest.raises(AssertionError, match="namespaces do not match"):
        func(xp.asarray(0), np.asarray(0))
    with pytest.raises(TypeError, match="Unrecognized array input"):
        func(xp.asarray(0), 0)
    with pytest.raises(TypeError, match="list is not a supported array type"):
        func(xp.asarray([0]), [0])


@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no isdtype")
def test_assert_close_tolerance(xp: ModuleType):
    xp_assert_close(xp.asarray([100.0]), xp.asarray([102.0]), rtol=0.03)
    with pytest.raises(AssertionError):
        xp_assert_close(xp.asarray([100.0]), xp.asarray([102.0]), rtol=0.01)

    xp_assert_close(xp.asarray([100.0]), xp.asarray([102.0]), atol=3)
    with pytest.raises(AssertionError):
        xp_assert_close(xp.asarray([100.0]), xp.asarray([102.0]), atol=1)


@param_assert_equal_close
@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no bool indexing")
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


def good_lazy(x: Array) -> Array:
    """A function that behaves well in dask and jax.jit"""
    return x * 2.0


def non_materializable(x: Array) -> Array:
    """
    This function materializes the input array, so it will fail when wrapped in jax.jit
    and it will trigger an expensive computation in dask.
    """
    xp = array_namespace(x)
    # On dask, this triggers two computations of the whole graph
    if xp.any(x < 0.0) or xp.any(x > 10.0):
        msg = "Values must be in the [0, 10] range"
        raise ValueError(msg)
    return x


def non_materializable2(x: Array) -> Array:
    return non_materializable(x)


def non_materializable3(x: Array) -> Array:
    return non_materializable(x)


def non_materializable4(x: Array) -> Array:
    return non_materializable(x)


lazy_xp_function(good_lazy)
# Works on JAX and Dask
lazy_xp_function(non_materializable2, jax_jit=False, allow_dask_compute=2)
# Works on JAX, but not Dask
lazy_xp_function(non_materializable3, jax_jit=False, allow_dask_compute=1)
# Works neither on Dask nor JAX
lazy_xp_function(non_materializable4)


def test_lazy_xp_function(xp: ModuleType):
    x = xp.asarray([1.0, 2.0])

    xp_assert_equal(good_lazy(x), xp.asarray([2.0, 4.0]))
    # Not wrapped
    xp_assert_equal(non_materializable(x), xp.asarray([1.0, 2.0]))
    # Wrapping explicitly disabled
    xp_assert_equal(non_materializable2(x), xp.asarray([1.0, 2.0]))

    if is_jax_namespace(xp):
        xp_assert_equal(non_materializable3(x), xp.asarray([1.0, 2.0]))
        with pytest.raises(
            TypeError, match="Attempted boolean conversion of traced array"
        ):
            non_materializable4(x)  # Wrapped

    elif is_dask_namespace(xp):
        with pytest.raises(
            AssertionError,
            match=r"dask\.compute.* 2 times, but only up to 1 calls are allowed",
        ):
            non_materializable3(x)
        with pytest.raises(
            AssertionError,
            match=r"dask\.compute.* 1 times, but no calls are allowed",
        ):
            non_materializable4(x)

    else:
        xp_assert_equal(non_materializable3(x), xp.asarray([1.0, 2.0]))
        xp_assert_equal(non_materializable4(x), xp.asarray([1.0, 2.0]))


def static_params(x: Array, n: int, flag: bool = False) -> Array:
    """Function with static parameters that must not be jitted"""
    if flag and n > 0:  # This fails if n or flag are jitted arrays
        return x * 2.0
    return x * 3.0


def static_params1(x: Array, n: int, flag: bool = False) -> Array:
    return static_params(x, n, flag)


def static_params2(x: Array, n: int, flag: bool = False) -> Array:
    return static_params(x, n, flag)


def static_params3(x: Array, n: int, flag: bool = False) -> Array:
    return static_params(x, n, flag)


lazy_xp_function(static_params1, static_argnums=(1, 2))
lazy_xp_function(static_params2, static_argnames=("n", "flag"))
lazy_xp_function(static_params3, static_argnums=1, static_argnames="flag")


@pytest.mark.parametrize("func", [static_params1, static_params2, static_params3])
def test_lazy_xp_function_static_params(xp: ModuleType, func: Callable[..., Array]):  # type: ignore[no-any-explicit]
    x = xp.asarray([1.0, 2.0])
    xp_assert_equal(func(x, 1), xp.asarray([3.0, 6.0]))
    xp_assert_equal(func(x, 1, True), xp.asarray([2.0, 4.0]))
    xp_assert_equal(func(x, 1, False), xp.asarray([3.0, 6.0]))
    xp_assert_equal(func(x, 0, False), xp.asarray([3.0, 6.0]))
    xp_assert_equal(func(x, 1, flag=True), xp.asarray([2.0, 4.0]))
    xp_assert_equal(func(x, n=1, flag=True), xp.asarray([2.0, 4.0]))
