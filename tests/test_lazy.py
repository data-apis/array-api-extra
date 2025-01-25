from types import ModuleType
from typing import NamedTuple

import numpy as np
import pytest

from array_api_extra import lazy_apply
from array_api_extra._lib import Backend
from array_api_extra._lib._testing import xp_assert_equal
from array_api_extra._lib._utils._typing import Array
from array_api_extra.testing import lazy_xp_function

skip_as_numpy = [
    pytest.mark.skip_xp_backend(Backend.CUPY, reason="device->host transfer"),
    pytest.mark.skip_xp_backend(Backend.SPARSE, reason="densification"),
]


@pytest.mark.parametrize("as_numpy", [False, pytest.param(True, marks=skip_as_numpy)])
def test_lazy_apply_kwargs(xp: ModuleType, library: Backend, as_numpy: bool) -> None:
    expect = np.ndarray if as_numpy or library is Backend.DASK else type(xp.asarray(0))

    class NT(NamedTuple):
        a: Array

    def f(
        x: Array,
        z: dict[str, list[Array] | tuple[Array, ...] | NT],
        msg: str,
        msgs: list[str],
    ) -> Array:
        assert isinstance(x, expect)
        assert isinstance(z["foo"], NT)
        assert isinstance(z["foo"].a, expect)
        assert isinstance(z["bar"][0], expect)
        assert isinstance(z["baz"][0], expect)
        assert msg == "Hello World"
        assert msgs[0] == "Hello World"
        return x

    x = xp.asarray(0)
    y = lazy_apply(  # pyright: ignore[reportCallIssue]
        f,
        x,
        z={"foo": NT(x), "bar": [x], "baz": (x,)},
        msg="Hello World",
        msgs=["Hello World"],
        shape=x.shape,
        dtype=x.dtype,
        as_numpy=as_numpy,
    )
    xp_assert_equal(x, y)


class CustomError(Exception):
    pass


def raises(x: Array) -> Array:
    def eager(_: Array) -> Array:
        msg = "Hello World"
        raise CustomError(msg)

    return lazy_apply(eager, x, shape=x.shape, dtype=x.dtype)


lazy_xp_function(raises)


def test_lazy_apply_raises(xp: ModuleType, library: Backend) -> None:
    x = xp.asarray(0)

    with pytest.raises(
        # FIXME https://github.com/jax-ml/jax/issues/26102
        RuntimeError if library is Backend.JAX else CustomError,
        match="Hello World",
    ):
        # Here we are disregarding the return value, which would
        # normally cause the graph not to materialize and the
        # exception not to be raised.
        # However, lazy_xp_function will do it for us on function exit.
        raises(x)
