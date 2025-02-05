from types import ModuleType
from typing import NamedTuple

import numpy as np
import pytest

import array_api_extra as xpx  # Let some tests bypass lazy_xp_function
from array_api_extra import lazy_apply
from array_api_extra._lib import Backend
from array_api_extra._lib._testing import xp_assert_equal
from array_api_extra._lib._utils import _compat
from array_api_extra._lib._utils._compat import array_namespace
from array_api_extra._lib._utils._typing import Array, Device
from array_api_extra.testing import lazy_xp_function

lazy_xp_function(
    lazy_apply, static_argnames=("func", "shape", "dtype", "as_numpy", "xp")
)

as_numpy = pytest.mark.parametrize(
    "as_numpy",
    [
        False,
        pytest.param(
            True,
            marks=[
                pytest.mark.skip_xp_backend(Backend.CUPY, reason="device->host copy"),
                pytest.mark.skip_xp_backend(Backend.SPARSE, reason="densification"),
            ],
        ),
    ],
)


@as_numpy
@pytest.mark.parametrize("shape", [(2,), (3, 2)])
@pytest.mark.parametrize("dtype", ["int32", "float64"])
def test_lazy_apply_simple(
    xp: ModuleType, library: Backend, shape: tuple[int, ...], dtype: str, as_numpy: bool
):
    def f(x: Array) -> Array:
        xp2 = array_namespace(x)
        if as_numpy or library in (Backend.NUMPY_READONLY, Backend.DASK):
            assert isinstance(x, np.ndarray)
        else:
            assert xp2 is xp

        y = xp2.broadcast_to(xp2.astype(x + 1, getattr(xp2, dtype)), shape)
        return xp2.asarray(y, copy=True)  # Torch: ensure writeable numpy array

    x = xp.asarray([1, 2], dtype=xp.int16)
    expect = xp.broadcast_to(xp.astype(x + 1, getattr(xp, dtype)), shape)
    actual = lazy_apply(f, x, shape=shape, dtype=getattr(xp, dtype), as_numpy=as_numpy)
    xp_assert_equal(actual, expect)


@as_numpy
def test_lazy_apply_broadcast(xp: ModuleType, as_numpy: bool):
    """Test that default shape and dtype are broadcasted from the inputs."""

    def f(x: Array, y: Array) -> Array:
        return x + y

    x = xp.asarray([1, 2], dtype=xp.int16)
    y = xp.asarray([[4], [5], [6]], dtype=xp.int32)
    z = lazy_apply(f, x, y, as_numpy=as_numpy)
    xp_assert_equal(z, x + y)


@as_numpy
def test_lazy_apply_multi_output(xp: ModuleType, as_numpy: bool):
    def f(x: Array) -> tuple[Array, Array]:
        xp2 = array_namespace(x)
        y = x + xp2.asarray(2, dtype=xp2.int8)  # Sparse: bad dtype propagation
        z = xp2.broadcast_to(xp2.astype(x + 1, xp2.int16), (3, 2))
        z = xp2.asarray(z, copy=True)  # Torch: ensure writeable numpy array
        return y, z

    x = xp.asarray([1, 2], dtype=xp.int8)
    expect = (
        xp.asarray([3, 4], dtype=xp.int8),
        xp.asarray([[2, 3], [2, 3], [2, 3]], dtype=xp.int16),
    )
    actual = lazy_apply(
        f, x, shape=((2,), (3, 2)), dtype=(xp.int8, xp.int16), as_numpy=as_numpy
    )
    assert isinstance(actual, tuple)
    assert len(actual) == 2
    xp_assert_equal(actual[0], expect[0])
    xp_assert_equal(actual[1], expect[1])


def test_lazy_apply_core_indices(da: ModuleType):
    """
    Test that a function that performs reductions along axes does so
    globally and not locally to each Dask chunk.
    """

    def f(x: Array) -> Array:
        return x.sum(axis=0) + x

    x_np = np.arange(15).reshape(5, 3)
    expect = da.asarray(f(x_np))
    x_da = da.asarray(x_np).rechunk(3)

    # A naive map_blocks fails because it applies f to each chunk separately,
    # but f needs to reduce along axis 0 which is broken into multiple chunks.
    # axis 0 is a "core axis" or "core index" (from xarray.apply_ufunc's
    # "core dimension").
    with pytest.raises(AssertionError):
        xp_assert_equal(da.map_blocks(f, x_da), expect)

    xp_assert_equal(lazy_apply(f, x_da), expect)


def test_lazy_apply_dont_run_on_meta(da: ModuleType):
    """Test that Dask won't try running func on the meta array,
    as it may have minimum size requirements.
    """

    def f(x: Array) -> Array:
        assert x.size
        return x + 1

    x = da.arange(10)
    assert not x._meta.size
    y = lazy_apply(f, x)
    xp_assert_equal(y, x + 1)


@pytest.mark.xfail_xp_backend(Backend.JAX, reason="unknown shape")
def test_lazy_apply_none_shape_in_args(xp: ModuleType, library: Backend):
    x = xp.asarray([1, 1, 2, 2, 2])

    xp2 = np if library is Backend.DASK else xp

    # Single output
    values = lazy_apply(xp2.unique_values, x, shape=(None,))
    xp_assert_equal(values, xp.asarray([1, 2]))

    # Multi output
    int_type = xp.asarray(0).dtype
    values, counts = lazy_apply(
        xp2.unique_counts,
        x,
        shape=((None,), (None,)),
        dtype=(x.dtype, int_type),
    )
    xp_assert_equal(values, xp.asarray([1, 2]))
    xp_assert_equal(counts, xp.asarray([2, 3]))


def check_lazy_apply_none_shape_broadcast(x: Array) -> Array:
    def f(x: Array) -> Array:
        return x

    x = x[x > 1]
    return lazy_apply(f, x)


lazy_xp_function(check_lazy_apply_none_shape_broadcast)


@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="bool mask")
@pytest.mark.xfail_xp_backend(Backend.JAX, reason="unknown shape")
def test_lazy_apply_none_shape_broadcast(xp: ModuleType):
    """Broadcast from input array with unknown shape"""
    x = xp.asarray([1, 2, 2])
    actual = check_lazy_apply_none_shape_broadcast(x)
    xp_assert_equal(actual, xp.asarray([2, 2]))


@pytest.mark.parametrize(
    "as_numpy",
    [
        False,
        pytest.param(
            True,
            marks=[
                pytest.mark.skip_xp_backend(
                    Backend.ARRAY_API_STRICT, reason="device->host copy"
                ),
                pytest.mark.skip_xp_backend(Backend.CUPY, reason="device->host copy"),
                pytest.mark.skip_xp_backend(Backend.SPARSE, reason="densification"),
            ],
        ),
    ],
)
def test_lazy_apply_device(xp: ModuleType, as_numpy: bool, device: Device):
    def f(x: Array) -> Array:
        xp2 = array_namespace(x)
        # Deliberately forgetting to add device here to test that the
        # output is transferred to the right device. This is necessary when
        # as_numpy=True anyway.
        return xp2.zeros(x.shape, dtype=x.dtype)

    x = xp.asarray([1, 2], device=device)
    y = lazy_apply(f, x, as_numpy=as_numpy)
    assert _compat.device(y) == device


def test_lazy_apply_no_args(xp: ModuleType):
    with pytest.raises(ValueError, match="at least one argument"):
        lazy_apply(lambda: xp.zeros(1), shape=(1,), dtype=xp.zeros(1).dtype, xp=xp)


class NT(NamedTuple):
    a: Array


def check_lazy_apply_kwargs(x: Array, expect_cls: type, as_numpy: bool) -> Array:
    def eager(
        x: Array,
        z: dict[str, list[Array] | tuple[Array, ...] | NT],
        msg: str,
        msgs: list[str],
        scalar: int,
    ) -> Array:
        assert isinstance(x, expect_cls)
        # JAX will crash if x isn't material
        assert int(x) == 0  # type: ignore[call-overload]
        # Did we re-wrap the namedtuple correctly, or did it get
        # accidentally changed to a basic tuple?
        assert isinstance(z["foo"], NT)
        assert isinstance(z["foo"].a, expect_cls)
        assert isinstance(z["bar"][0], expect_cls)  # list
        assert isinstance(z["baz"][0], expect_cls)  # tuple
        assert msg == "Hello World"  # must be hidden from JAX
        assert msgs[0] == "Hello World"  # must be hidden from JAX
        assert isinstance(msg, str)
        assert isinstance(msgs[0], str)
        assert scalar == 1  # must be hidden from JAX
        assert isinstance(scalar, int)
        return x + 1  # type: ignore[operator]

    # Use explicit namespace to bypass monkey-patching by lazy_xp_function
    return xpx.lazy_apply(  # pyright: ignore[reportCallIssue]
        eager,
        x,
        # These kwargs can and should be passed through jax.pure_callback
        z={"foo": NT(x), "bar": [x], "baz": (x,)},
        # These can't
        msg="Hello World",
        msgs=["Hello World"],
        # This will be automatically cast to jax.Array if we don't wrap it
        scalar=1,
        shape=x.shape,
        dtype=x.dtype,
        as_numpy=as_numpy,
    )


lazy_xp_function(check_lazy_apply_kwargs, static_argnames=("expect_cls", "as_numpy"))


@as_numpy
def test_lazy_apply_kwargs(xp: ModuleType, library: Backend, as_numpy: bool) -> None:
    """When as_numpy=True, search and replace arrays in the (nested) keywords arguments
    with numpy arrays, and leave the rest untouched."""
    expect_cls = (
        np.ndarray if as_numpy or library is Backend.DASK else type(xp.asarray(0))
    )
    x = xp.asarray(0)
    actual = check_lazy_apply_kwargs(x, expect_cls, as_numpy)  # pyright: ignore[reportUnknownArgumentType]
    xp_assert_equal(actual, x + 1)


class CustomError(Exception):
    pass


def raises(x: Array) -> Array:
    def eager(_: Array) -> Array:
        msg = "Hello World"
        raise CustomError(msg)

    # Use explicit namespace to bypass monkey-patching by lazy_xp_function
    return xpx.lazy_apply(eager, x, shape=x.shape, dtype=x.dtype)


# jax.pure_callback does not support raising
# https://github.com/jax-ml/jax/issues/26102
lazy_xp_function(raises, jax_jit=False)


def test_lazy_apply_raises(xp: ModuleType) -> None:
    x = xp.asarray(0)

    with pytest.raises(CustomError, match="Hello World"):
        # Here we are disregarding the return value, which would
        # normally cause the graph not to materialize and the
        # exception not to be raised.
        # However, lazy_xp_function will do it for us on function exit.
        raises(x)
