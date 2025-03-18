from collections.abc import Callable
from types import ModuleType
from typing import cast

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

# mypy: disable-error-code=decorated-any
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
def test_assert_close_equal_basic(xp: ModuleType, func: Callable[..., None]):  # type: ignore[explicit-any]
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
def test_assert_close_equal_namespace(xp: ModuleType, func: Callable[..., None]):  # type: ignore[explicit-any]
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
@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="index by sparse array")
def test_assert_close_equal_none_shape(xp: ModuleType, func: Callable[..., None]):  # type: ignore[explicit-any]
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
    # Crashes inside jax.jit
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
            _ = non_materializable4(x)  # Wrapped

    elif is_dask_namespace(xp):
        with pytest.raises(
            AssertionError,
            match=r"dask\.compute.* 2 times, but only up to 1 calls are allowed",
        ):
            _ = non_materializable3(x)
        with pytest.raises(
            AssertionError,
            match=r"dask\.compute.* 1 times, but no calls are allowed",
        ):
            _ = non_materializable4(x)

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
def test_lazy_xp_function_static_params(xp: ModuleType, func: Callable[..., Array]):  # type: ignore[explicit-any]
    x = xp.asarray([1.0, 2.0])
    xp_assert_equal(func(x, 1), xp.asarray([3.0, 6.0]))
    xp_assert_equal(func(x, 1, True), xp.asarray([2.0, 4.0]))
    xp_assert_equal(func(x, 1, False), xp.asarray([3.0, 6.0]))
    xp_assert_equal(func(x, 0, False), xp.asarray([3.0, 6.0]))
    xp_assert_equal(func(x, 1, flag=True), xp.asarray([2.0, 4.0]))
    xp_assert_equal(func(x, n=1, flag=True), xp.asarray([2.0, 4.0]))


try:
    # Test an arbitrary Cython ufunc (@cython.vectorize).
    # When SCIPY_ARRAY_API is not set, this is the same as
    # scipy.special.erf.
    from scipy.special._ufuncs import erf  # type: ignore[import-not-found]

    lazy_xp_function(erf)  # pyright: ignore[reportUnknownArgumentType]
except ImportError:
    erf = None


@pytest.mark.filterwarnings("ignore:__array_wrap__:DeprecationWarning")  # torch
def test_lazy_xp_function_cython_ufuncs(xp: ModuleType, library: Backend):
    pytest.importorskip("scipy")
    assert erf is not None
    x = xp.asarray([6.0, 7.0])
    if library in (Backend.ARRAY_API_STRICT, Backend.JAX):
        # array-api-strict arrays are auto-converted to numpy
        # which results in an assertion error for mismatched namespaces
        # eager jax arrays are auto-converted to numpy in eager jax
        # and fail in jax.jit (which lazy_xp_function tests here)
        with pytest.raises((TypeError, AssertionError)):
            xp_assert_equal(cast(Array, erf(x)), xp.asarray([1.0, 1.0]))
    else:
        # cupy, dask and sparse define __array_ufunc__ and dispatch accordingly
        # note that when sparse reduces to scalar it returns a np.generic, which
        # would make xp_assert_equal fail.
        xp_assert_equal(cast(Array, erf(x)), xp.asarray([1.0, 1.0]))


def dask_raises(x: Array) -> Array:
    def _raises(x: Array) -> Array:
        # Test that map_blocks doesn't eagerly call the function;
        # dtype and meta should be sufficient to skip the trial run.
        assert x.shape == (3,)
        msg = "Hello world"
        raise ValueError(msg)

    return x.map_blocks(_raises, dtype=x.dtype, meta=x._meta)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]


lazy_xp_function(dask_raises)


def test_lazy_xp_function_eagerly_raises(da: ModuleType):
    """Test that the pattern::

        with pytest.raises(Exception):
            func(x)

    works with Dask, even though it normally wouldn't as we're disregarding the func
    output so the graph would not be ordinarily materialized.
    lazy_xp_function contains ad-hoc code to materialize and reraise exceptions.
    """
    x = da.arange(3)
    with pytest.raises(ValueError, match="Hello world"):
        _ = dask_raises(x)


wrapped = ModuleType("wrapped")
naked = ModuleType("naked")


def f(x: Array) -> Array:
    xp = array_namespace(x)
    # Crash in jax.jit and trigger compute() on dask
    if not xp.all(x):
        msg = "Values must be non-zero"
        raise ValueError(msg)
    return x


wrapped.f = f  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
naked.f = f  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
del f


lazy_xp_function(wrapped.f)
lazy_xp_modules = [wrapped]


def test_lazy_xp_modules(xp: ModuleType, library: Backend):
    x = xp.asarray([1.0, 2.0])
    y = naked.f(x)
    xp_assert_equal(y, x)

    if library is Backend.JAX:
        with pytest.raises(
            TypeError, match="Attempted boolean conversion of traced array"
        ):
            wrapped.f(x)
    elif library is Backend.DASK:
        with pytest.raises(AssertionError, match=r"dask\.compute"):
            wrapped.f(x)
    else:
        y = wrapped.f(x)
        xp_assert_equal(y, x)
