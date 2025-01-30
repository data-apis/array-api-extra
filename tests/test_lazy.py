from types import ModuleType
from typing import NamedTuple

import numpy as np
import pytest

from array_api_extra import lazy_apply
from array_api_extra._lib import Backend
from array_api_extra._lib._testing import xp_assert_equal
from array_api_extra._lib._utils._typing import Array
from array_api_extra.testing import lazy_xp_function

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
def test_lazy_apply_simple(xp: ModuleType, as_numpy: bool):
    pytest.skip("TODO")


@as_numpy
def test_lazy_apply_broadcast(xp: ModuleType, as_numpy: bool):
    pytest.skip("TODO")


@as_numpy
def test_lazy_apply_multi_output(xp: ModuleType, as_numpy: bool):
    pytest.skip("TODO")


def test_lazy_apply_core_indices(da: ModuleType):
    """Test that a func that performs reductions along axes does so
    globally and not locally to each Dask chunk.
    """
    pytest.skip("TODO")


def test_lazy_apply_dont_run_on_meta(da: ModuleType):
    """Test that Dask won't try running func on the meta array,
    as it may have minimum size requirements.
    """
    pytest.skip("TODO")


def test_lazy_apply_none_shape(da: ModuleType):
    pytest.skip("TODO")


@as_numpy
def test_lazy_apply_device(xp: ModuleType, as_numpy: bool):
    pytest.skip("TODO")


@as_numpy
def test_lazy_apply_no_args(xp: ModuleType, as_numpy: bool):
    pytest.skip("TODO")


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
        assert int(x) == 0  # JAX will crash if x isn't material
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

    return lazy_apply(  # pyright: ignore[reportCallIssue]
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

    return lazy_apply(eager, x, shape=x.shape, dtype=x.dtype)


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
