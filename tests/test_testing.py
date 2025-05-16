from collections.abc import Callable
from contextlib import nullcontext
from types import ModuleType
from typing import cast

import numpy as np
import pytest

from array_api_extra._lib._backends import Backend
from array_api_extra._lib._testing import (
    as_numpy_array,
    xp_assert_close,
    xp_assert_equal,
    xp_assert_less,
)
from array_api_extra._lib._utils._compat import (
    array_namespace,
    is_dask_namespace,
    is_jax_namespace,
)
from array_api_extra._lib._utils._typing import Array, Device
from array_api_extra.testing import lazy_xp_function

# mypy: disable-error-code=decorated-any
# pyright: reportUnknownParameterType=false,reportMissingParameterType=false

param_assert_equal_close = pytest.mark.parametrize(
    "func",
    [
        xp_assert_equal,
        xp_assert_less,
        pytest.param(
            xp_assert_close,
            marks=pytest.mark.xfail_xp_backend(
                Backend.SPARSE, reason="no isdtype", strict=False
            ),
        ),
    ],
)


def test_as_numpy_array(xp: ModuleType, device: Device):
    x = xp.asarray([1, 2, 3], device=device)
    y = as_numpy_array(x, xp=xp)
    assert isinstance(y, np.ndarray)


@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no isdtype", strict=False)
@pytest.mark.parametrize("func", [xp_assert_equal, xp_assert_close])
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
@pytest.mark.parametrize("func", [xp_assert_equal, xp_assert_close, xp_assert_less])
def test_assert_close_equal_less_namespace(xp: ModuleType, func: Callable[..., None]):  # type: ignore[explicit-any]
    with pytest.raises(AssertionError, match="namespaces do not match"):
        func(xp.asarray(0), np.asarray(0))
    with pytest.raises(TypeError, match="Unrecognized array input"):
        func(xp.asarray(0), 0)
    with pytest.raises(TypeError, match="list is not a supported array type"):
        func(xp.asarray([0]), [0])


@param_assert_equal_close
@pytest.mark.parametrize("check_shape", [False, True])
def test_assert_close_equal_less_shape(  # type: ignore[explicit-any]
    xp: ModuleType,
    func: Callable[..., None],
    check_shape: bool,
):
    context = (
        pytest.raises(AssertionError, match="shapes do not match")
        if check_shape
        else nullcontext()
    )
    with context:
        # note: NaNs are handled by all 3 checks
        func(xp.asarray([xp.nan, xp.nan]), xp.asarray(xp.nan), check_shape=check_shape)


@param_assert_equal_close
@pytest.mark.parametrize("check_dtype", [False, True])
def test_assert_close_equal_less_dtype(  # type: ignore[explicit-any]
    xp: ModuleType,
    func: Callable[..., None],
    check_dtype: bool,
):
    context = (
        pytest.raises(AssertionError, match="dtypes do not match")
        if check_dtype
        else nullcontext()
    )
    with context:
        func(
            xp.asarray(xp.nan, dtype=xp.float32),
            xp.asarray(xp.nan, dtype=xp.float64),
            check_dtype=check_dtype,
        )


@pytest.mark.parametrize("func", [xp_assert_equal, xp_assert_close, xp_assert_less])
@pytest.mark.parametrize("check_scalar", [False, True])
def test_assert_close_equal_less_scalar(  # type: ignore[explicit-any]
    xp: ModuleType,
    func: Callable[..., None],
    check_scalar: bool,
):
    context = (
        pytest.raises(AssertionError, match="array-ness does not match")
        if check_scalar
        else nullcontext()
    )
    with context:
        func(np.asarray(xp.nan), np.asarray(xp.nan)[()], check_scalar=check_scalar)


@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no isdtype")
def test_assert_close_tolerance(xp: ModuleType):
    xp_assert_close(xp.asarray([100.0]), xp.asarray([102.0]), rtol=0.03)
    with pytest.raises(AssertionError):
        xp_assert_close(xp.asarray([100.0]), xp.asarray([102.0]), rtol=0.01)

    xp_assert_close(xp.asarray([100.0]), xp.asarray([102.0]), atol=3)
    with pytest.raises(AssertionError):
        xp_assert_close(xp.asarray([100.0]), xp.asarray([102.0]), atol=1)


def test_assert_less_basic(xp: ModuleType):
    xp_assert_less(xp.asarray(-1), xp.asarray(0))
    xp_assert_less(xp.asarray([1, 2]), xp.asarray([2, 3]))
    with pytest.raises(AssertionError):
        xp_assert_less(xp.asarray([1, 1]), xp.asarray([2, 1]))
    with pytest.raises(AssertionError, match="hello"):
        xp_assert_less(xp.asarray([1, 1]), xp.asarray([2, 1]), err_msg="hello")


@pytest.mark.skip_xp_backend(Backend.SPARSE, reason="index by sparse array")
@pytest.mark.skip_xp_backend(Backend.ARRAY_API_STRICTEST, reason="boolean indexing")
@pytest.mark.parametrize("func", [xp_assert_equal, xp_assert_close])
def test_assert_close_equal_none_shape(xp: ModuleType, func: Callable[..., None]):  # type: ignore[explicit-any]
    """On Dask and other lazy backends, test that a shape with NaN's or None's
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
    """A function that behaves well in Dask and jax.jit"""
    return x * 2.0


def non_materializable(x: Array) -> Array:
    """
    This function materializes the input array, so it will fail when wrapped in jax.jit
    and it will trigger an expensive computation in Dask.
    """
    xp = array_namespace(x)
    # Crashes inside jax.jit
    # On Dask, this triggers two computations of the whole graph
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


def non_materializable5(x: Array) -> Array:
    return non_materializable(x)


lazy_xp_function(good_lazy)
# Works on JAX and Dask
lazy_xp_function(non_materializable2, jax_jit=False, allow_dask_compute=2)
lazy_xp_function(non_materializable3, jax_jit=False, allow_dask_compute=True)
# Works on JAX, but not Dask
lazy_xp_function(non_materializable4, jax_jit=False, allow_dask_compute=1)
# Works neither on Dask nor JAX
lazy_xp_function(non_materializable5)


def test_lazy_xp_function(xp: ModuleType):
    x = xp.asarray([1.0, 2.0])

    xp_assert_equal(good_lazy(x), xp.asarray([2.0, 4.0]))
    # Not wrapped
    xp_assert_equal(non_materializable(x), xp.asarray([1.0, 2.0]))
    # Wrapping explicitly disabled
    xp_assert_equal(non_materializable2(x), xp.asarray([1.0, 2.0]))
    xp_assert_equal(non_materializable3(x), xp.asarray([1.0, 2.0]))

    if is_jax_namespace(xp):
        xp_assert_equal(non_materializable4(x), xp.asarray([1.0, 2.0]))
        with pytest.raises(
            TypeError, match="Attempted boolean conversion of traced array"
        ):
            _ = non_materializable5(x)  # Wrapped

    elif is_dask_namespace(xp):
        with pytest.raises(
            AssertionError,
            match=r"dask\.compute.* 2 times, but only up to 1 calls are allowed",
        ):
            _ = non_materializable4(x)
        with pytest.raises(
            AssertionError,
            match=r"dask\.compute.* 1 times, but no calls are allowed",
        ):
            _ = non_materializable5(x)

    else:
        xp_assert_equal(non_materializable4(x), xp.asarray([1.0, 2.0]))
        xp_assert_equal(non_materializable5(x), xp.asarray([1.0, 2.0]))


def static_params(x: Array, n: int, flag: bool = False) -> Array:
    """Function with static parameters that must not be jitted"""
    if flag and n > 0:  # This fails if n or flag are jitted arrays
        return x * 2.0
    return x * 3.0


lazy_xp_function(static_params)


def test_lazy_xp_function_static_params(xp: ModuleType):
    x = xp.asarray([1.0, 2.0])
    xp_assert_equal(static_params(x, 1), xp.asarray([3.0, 6.0]))
    xp_assert_equal(static_params(x, 1, True), xp.asarray([2.0, 4.0]))
    xp_assert_equal(static_params(x, 1, False), xp.asarray([3.0, 6.0]))
    xp_assert_equal(static_params(x, 0, False), xp.asarray([3.0, 6.0]))
    xp_assert_equal(static_params(x, 1, flag=True), xp.asarray([2.0, 4.0]))
    xp_assert_equal(static_params(x, n=1, flag=True), xp.asarray([2.0, 4.0]))


def test_lazy_xp_function_deprecated_static_argnames():
    with pytest.warns(DeprecationWarning, match="static_argnames"):
        lazy_xp_function(static_params, static_argnames=["flag"])  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    with pytest.warns(DeprecationWarning, match="static_argnums"):
        lazy_xp_function(static_params, static_argnums=[1])  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


try:
    # Test an arbitrary Cython ufunc (@cython.vectorize).
    # When SCIPY_ARRAY_API is not set, this is the same as
    # scipy.special.erf.
    from scipy.special._ufuncs import erf  # type: ignore[import-not-found]

    lazy_xp_function(erf)  # pyright: ignore[reportUnknownArgumentType]
except ImportError:
    erf = None


@pytest.mark.skip_xp_backend(Backend.TORCH_GPU, reason="device->host copy")
@pytest.mark.filterwarnings("ignore:__array_wrap__:DeprecationWarning")  # PyTorch
def test_lazy_xp_function_cython_ufuncs(xp: ModuleType, library: Backend):
    pytest.importorskip("scipy")
    assert erf is not None
    x = xp.asarray([6.0, 7.0])
    if library.like(Backend.ARRAY_API_STRICT, Backend.JAX):
        # array-api-strict arrays are auto-converted to NumPy
        # which results in an assertion error for mismatched namespaces
        # eager JAX arrays are auto-converted to NumPy in eager JAX
        # and fail in jax.jit (which lazy_xp_function tests here)
        with pytest.raises((TypeError, AssertionError)):
            xp_assert_equal(cast(Array, erf(x)), xp.asarray([1.0, 1.0]))
    else:
        # CuPy, Dask and sparse define __array_ufunc__ and dispatch accordingly
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


class Wrapper:
    """Trivial opaque wrapper. Must be pickleable."""

    x: Array

    def __init__(self, x: Array):
        self.x = x


def check_opaque_wrapper(w: Wrapper, xp: ModuleType) -> Wrapper:
    assert isinstance(w, Wrapper)
    assert array_namespace(w.x) == xp
    return Wrapper(w.x + 1)


lazy_xp_function(check_opaque_wrapper)


def test_lazy_xp_function_opaque_wrappers(xp: ModuleType):
    """
    Test that function input and output can be wrapped into arbitrary
    serializable Python objects, even if jax.jit does not support them.
    """
    x = xp.asarray([1, 2])
    xp2 = array_namespace(x)  # Revert NUMPY_READONLY to array_api_compat.numpy
    res = check_opaque_wrapper(Wrapper(x), xp2)
    xp_assert_equal(res.x, xp.asarray([2, 3]))


def test_lazy_xp_function_opaque_wrappers_eagerly_raise(da: ModuleType):
    """
    Like `test_lazy_xp_function_eagerly_raises`, but the returned object is
    wrapped in an opaque wrapper.
    """
    x = da.arange(3)
    with pytest.raises(ValueError, match="Hello world"):
        _ = Wrapper(dask_raises(x))


def check_recursive(x: list[object]) -> list[object]:
    assert isinstance(x, list)
    assert x[1] is x
    y: list[object] = [cast(Array, x[0]) + 1]
    y.append(y)
    return y


lazy_xp_function(check_recursive)


def test_lazy_xp_function_recursive(xp: ModuleType):
    """Test that inputs and outputs can be recursive data structures."""
    x: list[object] = [xp.asarray([1, 2])]
    x.append(x)
    y = check_recursive(x)
    assert isinstance(y, list)
    xp_assert_equal(cast(Array, y[0]), xp.asarray([2, 3]))
    assert y[1] is y


wrapped = ModuleType("wrapped")
naked = ModuleType("naked")


def f(x: Array) -> Array:
    xp = array_namespace(x)
    # Crash in jax.jit and trigger compute() on Dask
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

    if library.like(Backend.JAX):
        with pytest.raises(
            TypeError, match="Attempted boolean conversion of traced array"
        ):
            wrapped.f(x)
    elif library.like(Backend.DASK):
        with pytest.raises(AssertionError, match=r"dask\.compute"):
            wrapped.f(x)
    else:
        y = wrapped.f(x)
        xp_assert_equal(y, x)
