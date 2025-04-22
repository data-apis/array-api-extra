import contextlib
from types import ModuleType
from typing import cast

import numpy as np
import pytest

import array_api_extra as xpx  # Let some tests bypass lazy_xp_function
from array_api_extra import lazy_apply
from array_api_extra._lib._backends import Backend
from array_api_extra._lib._testing import xp_assert_equal
from array_api_extra._lib._utils import _compat
from array_api_extra._lib._utils._compat import array_namespace, is_dask_array
from array_api_extra._lib._utils._helpers import eager_shape
from array_api_extra._lib._utils._typing import Array, Device
from array_api_extra.testing import lazy_xp_function

lazy_xp_function(lazy_apply)

as_numpy = pytest.mark.parametrize(
    "as_numpy",
    [
        False,
        pytest.param(
            True,
            marks=[
                pytest.mark.skip_xp_backend(Backend.CUPY, reason="device->host copy"),
                pytest.mark.skip_xp_backend(
                    Backend.TORCH_GPU, reason="device->host copy"
                ),
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
        return xp2.asarray(y, copy=True)  # PyTorch: ensure writeable NumPy array

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
        z = xp2.asarray(z, copy=True)  # PyTorch: ensure writeable NumPy array
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


@pytest.mark.parametrize(
    "as_numpy",
    [
        False,
        pytest.param(
            True,
            marks=[
                pytest.mark.skip_xp_backend(Backend.CUPY, reason="device->host copy"),
                pytest.mark.skip_xp_backend(
                    Backend.TORCH_GPU, reason="device->host copy"
                ),
                pytest.mark.skip_xp_backend(Backend.SPARSE, reason="densification"),
            ],
        ),
    ],
)
def test_lazy_apply_multi_output_broadcast_dtype(xp: ModuleType, as_numpy: bool):
    """
    If dtype is omitted and there are multiple shapes, use the same
    dtype for all output arrays, broadcasted from the inputs
    """

    def f(x: Array, y: Array) -> tuple[Array, Array]:
        return x + y, x - y

    x = xp.asarray([1, 2], dtype=xp.float32)
    y = xp.asarray([3], dtype=xp.float64)
    expect = (
        xp.asarray([4, 5], dtype=xp.float64),
        xp.asarray([-2, -1], dtype=xp.float64),
    )
    actual = lazy_apply(f, x, y, shape=((2,), (2,)), as_numpy=as_numpy)
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
        xp = array_namespace(x)
        return xp.sum(x, axis=0) + x

    x_np = cast(Array, np.arange(15).reshape(5, 3))  # type: ignore[bad-cast]
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


def test_lazy_apply_dask_non_numpy_meta(da: ModuleType):
    """Test Dask wrapping around a meta-namespace other than numpy."""
    # At the moment of writing, of all Array API namespaces CuPy is
    # the only one that Dask supports.
    # For this reason, we can only test as_numpy=False since
    # np.asarray(cp.Array) is blocked by the transfer guard.

    cp = pytest.importorskip("cupy")
    cp = array_namespace(cp.empty(0))
    x_cp = cp.asarray([1, 2, 3])
    x_da = da.asarray([1, 2, 3]).map_blocks(cp.asarray)
    assert array_namespace(x_da._meta) is cp

    def f(x: Array) -> Array:
        return x + 1

    y = lazy_apply(f, x_da)
    assert array_namespace(y._meta) is cp  # type: ignore[attr-defined]  # pyright: ignore[reportUnknownArgumentType,reportAttributeAccessIssue]
    xp_assert_equal(y.compute(), x_cp + 1)  # type: ignore[attr-defined]  # pyright: ignore[reportUnknownArgumentType,reportAttributeAccessIssue]


def test_dask_key(da: ModuleType):
    """Test that the function name is visible on the Dask dashboard and in metrics."""

    def helloworld(x: Array) -> Array:
        return x + 1

    x = da.asarray([1, 2])
    # Use full namespace to bypass monkey-patching by lazy_xp_function,
    # which calls persist() to materialize exceptions and warnings and in
    # doing so squashes the graph.
    y = xpx.lazy_apply(helloworld, x)

    prefixes = set()
    for key in y.__dask_graph__():  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
        name = key[0] if isinstance(key, tuple) else key
        assert isinstance(name, str)
        prefixes.add(name.split("-")[0])

    assert "helloworld" in prefixes


def test_lazy_apply_none_shape_in_args(xp: ModuleType, library: Backend):
    x = xp.asarray([1, 1, 2, 2, 2])

    # TODO mxp = meta_namespace(x, xp=xp)
    mxp = np if library is Backend.DASK else xp
    int_type = xp.asarray(0).dtype

    ctx: contextlib.AbstractContextManager[object]
    if library.like(Backend.JAX):
        ctx = pytest.raises(ValueError, match="Output shape must be fully known")
    elif library is Backend.ARRAY_API_STRICTEST:
        ctx = pytest.raises(RuntimeError, match="data-dependent shapes")
    else:
        ctx = contextlib.nullcontext()

    # Single output
    with ctx:
        values = lazy_apply(mxp.unique_values, x, shape=(None,))
        xp_assert_equal(values, xp.asarray([1, 2]))

    with ctx:
        # Multi output
        values, counts = lazy_apply(
            mxp.unique_counts,
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
    # Use explicit namespace to bypass monkey-patching by lazy_xp_function
    return xpx.lazy_apply(f, x)


lazy_xp_function(check_lazy_apply_none_shape_broadcast)


@pytest.mark.skip_xp_backend(Backend.SPARSE, reason="index by sparse array")
@pytest.mark.skip_xp_backend(Backend.JAX, reason="boolean indexing")
@pytest.mark.skip_xp_backend(Backend.JAX_GPU, reason="boolean indexing")
@pytest.mark.skip_xp_backend(Backend.ARRAY_API_STRICTEST, reason="boolean indexing")
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
                pytest.mark.skip_xp_backend(
                    Backend.TORCH_GPU, reason="device->host copy"
                ),
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


def test_lazy_apply_arraylike(xp: ModuleType):
    """Wrapped func returns an array-like"""
    x = xp.asarray([1, 2, 3])

    # Single output
    def f(x: Array) -> int:
        shape = eager_shape(x)
        return shape[0]

    expect = xp.asarray(3)
    actual = lazy_apply(f, x, shape=(), dtype=expect.dtype)
    xp_assert_equal(actual, expect)

    # Multi output
    def g(x: Array) -> tuple[int, list[int]]:
        shape = eager_shape(x)
        return shape[0], list(shape)

    actual2 = lazy_apply(g, x, shape=((), (1,)), dtype=(expect.dtype, expect.dtype))
    xp_assert_equal(actual2[0], xp.asarray(3))
    xp_assert_equal(actual2[1], xp.asarray([3]))


def test_lazy_apply_scalars_and_nones(xp: ModuleType, library: Backend):
    def f(x: Array, y: None, z: int | Array) -> Array:
        mxp = array_namespace(x, y, z)
        mtyp = type(mxp.asarray(0))
        assert isinstance(x, mtyp)
        assert y is None
        # jax.pure_callback wraps scalar args
        assert isinstance(z, mtyp if library.like(Backend.JAX) else int)
        return x + z

    x = xp.asarray([1, 2])
    w = lazy_apply(f, x, None, 3)
    xp_assert_equal(w, x + 3)


def check_lazy_apply_kwargs(x: Array, expect_cls: type, as_numpy: bool) -> Array:
    is_dask = is_dask_array(x)
    recursive: list[object] = []
    if not is_dask:  # dask.delayed crashes on recursion
        recursive.append(recursive)

    def eager(
        x: Array,
        z: dict[int, list[int]],
        msg: str,
        msgs: list[str],
        scalar: int,
        recursive: list[list[object]],
    ) -> Array:
        assert isinstance(x, expect_cls)
        # JAX will crash if x isn't material
        assert int(x) == 0
        # Did we re-wrap the namedtuple correctly, or did it get
        # accidentally changed to a basic tuple?
        assert z == {0: [1, 2]}
        assert msg == "Hello World"  # must be hidden from JAX
        assert msgs[0] == "Hello World"  # must be hidden from JAX
        assert isinstance(msg, str)
        assert isinstance(msgs[0], str)
        assert scalar == 1  # must be hidden from JAX
        assert isinstance(scalar, int)
        assert isinstance(recursive, list)
        if not is_dask:
            assert recursive[0][0] is recursive[0]
        return x + 1

    # Use explicit namespace to bypass monkey-patching by lazy_xp_function
    return xpx.lazy_apply(  # pyright: ignore[reportCallIssue]
        eager,
        x,
        z={0: [1, 2]},
        msg="Hello World",
        msgs=["Hello World"],
        # This will be automatically cast to jax.Array if we don't wrap it
        scalar=1,
        recursive=recursive,
        shape=x.shape,
        dtype=x.dtype,
        as_numpy=as_numpy,
    )


lazy_xp_function(check_lazy_apply_kwargs)


@as_numpy
def test_lazy_apply_kwargs(xp: ModuleType, library: Backend, as_numpy: bool):
    """When as_numpy=True, search and replace arrays in the (nested) keywords arguments
    with numpy arrays, and leave the rest untouched."""
    x = xp.asarray(0)
    expect_cls = np.ndarray if as_numpy or library is Backend.DASK else type(x)
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


def test_lazy_apply_raises(xp: ModuleType):
    """
    See Also
    --------
    test_testing.py::test_lazy_xp_function_eagerly_raises
    """
    x = xp.asarray(0)

    with pytest.raises(CustomError, match="Hello World"):
        # Here we are disregarding the return value, which would
        # normally cause the graph not to materialize and the
        # exception not to be raised.
        # However, lazy_xp_function will do it for us on function exit.
        _ = raises(x)


def test_invalid_args():
    def f(x: Array) -> Array:
        return x

    x = np.asarray(1)

    with pytest.raises(ValueError, match="at least one argument array"):
        _ = lazy_apply(f, shape=(1,), dtype=np.int32, xp=np)
    with pytest.raises(ValueError, match="at least one argument array"):
        _ = lazy_apply(f, 1, shape=(1,), dtype=np.int32, xp=np)
    with pytest.raises(ValueError, match="at least one argument array"):
        _ = lazy_apply(f, shape=(1,), dtype=np.int32)
    with pytest.raises(ValueError, match="multiple shapes but only one dtype"):
        _ = lazy_apply(f, x, shape=[(1,), (2,)], dtype=np.int32)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue,reportArgumentType]
    with pytest.raises(ValueError, match="single shape but multiple dtypes"):
        _ = lazy_apply(f, x, shape=(1,), dtype=[np.int32, np.int64])  # pyright: ignore[reportCallIssue,reportArgumentType]
    with pytest.raises(ValueError, match="2 shapes and 1 dtypes"):
        _ = lazy_apply(f, x, shape=[(1,), (2,)], dtype=[np.int32])  # type: ignore[arg-type]  # pyright: ignore[reportCallIssue,reportArgumentType]
