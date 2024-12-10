from __future__ import annotations

from contextlib import contextmanager, suppress
from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np
import pytest
from array_api_compat import (  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
    array_namespace,
    is_dask_array,
    is_pydata_sparse_array,
    is_writeable_array,
)

from array_api_extra import at

if TYPE_CHECKING:
    from array_api_extra._lib._typing import Array, Untyped

all_libraries = (
    "array_api_strict",
    "numpy",
    "numpy_readonly",
    "cupy",
    "torch",
    "dask.array",
    "sparse",
    "jax.numpy",
)


@pytest.fixture(params=all_libraries)
def array(request: pytest.FixtureRequest) -> Array:
    library = request.param
    if library == "numpy_readonly":
        x = np.asarray([10.0, 20.0, 30.0])
        x.flags.writeable = False
    else:
        try:
            lib = import_module(library)
        except ImportError:
            pytest.skip(f"{library} is not installed")
        x = lib.asarray([10.0, 20.0, 30.0])
    return x


def assert_array_equal(a: Array, b: Array) -> None:
    xp = array_namespace(a)
    b = xp.asarray(b)
    eq = xp.all(a == b)
    if is_dask_array(a):
        eq = eq.compute()
    assert eq


@contextmanager
def assert_copy(array: Array, copy: bool | None) -> Untyped:  # type: ignore[no-any-decorated]
    # dask arrays are writeable, but writing to them will hot-swap the
    # dask graph inside the collection so that anything that references
    # the original graph, i.e. the input collection, won't be mutated.
    if copy is False and not is_writeable_array(array):
        with pytest.raises((TypeError, ValueError)):
            yield
        return

    xp = array_namespace(array)
    array_orig = xp.asarray(array, copy=True)
    yield

    expect_copy = not is_writeable_array(array) if copy is None else copy
    assert_array_equal(xp.all(array == array_orig), expect_copy)


@pytest.mark.parametrize("copy", [True, False, None])
@pytest.mark.parametrize(
    ("op", "arg", "expect"),
    [
        ("set", 40.0, [10.0, 40.0, 40.0]),
        ("add", 40.0, [10.0, 60.0, 70.0]),
        ("subtract", 100.0, [10.0, -80.0, -70.0]),
        ("multiply", 2.0, [10.0, 40.0, 60.0]),
        ("divide", 2.0, [10.0, 10.0, 15.0]),
        ("power", 2.0, [10.0, 400.0, 900.0]),
        ("min", 25.0, [10.0, 20.0, 25.0]),
        ("max", 25.0, [10.0, 25.0, 30.0]),
    ],
)
def test_update_ops(
    array: Array, copy: bool | None, op: str, arg: float, expect: list[float]
):
    if is_pydata_sparse_array(array):
        pytest.skip("at() does not support updates on sparse arrays")

    with assert_copy(array, copy):
        y = getattr(at(array, slice(1, None)), op)(arg, copy=copy)
        assert isinstance(y, type(array))
        assert_array_equal(y, expect)


@pytest.mark.parametrize("copy", [True, False, None])
def test_get(array: Array, copy: bool | None):
    expect_copy = copy

    # dask is mutable, but __getitem__ never returns a view
    if is_dask_array(array):
        if copy is False:
            with pytest.raises(ValueError, match="always returns a copy"):
                at(array, slice(2)).get(copy=False)
            return
        expect_copy = True

    with assert_copy(array, expect_copy):
        y = at(array, slice(2)).get(copy=copy)
        assert isinstance(y, type(array))
        assert_array_equal(y, [10.0, 20.0])
        # Let assert_copy test that y is a view or copy
        with suppress(TypeError, ValueError):
            y[:] = 40


def test_get_bool_indices(array: Array):
    """get() with a boolean array index always returns a copy"""
    # sparse violates the array API as it doesn't support
    # a boolean index that is another sparse array.
    # dask with dask index has NaN size, which complicates testing.
    if is_pydata_sparse_array(array) or is_dask_array(array):
        xp = np
    else:
        xp = array_namespace(array)
    idx = xp.asarray([True, False, True])

    with pytest.raises(ValueError, match="copy"):
        at(array, idx).get(copy=False)

    assert_array_equal(at(array, idx).get(), [10.0, 30.0])

    with assert_copy(array, True):
        y = at(array, idx).get(copy=True)
        assert_array_equal(y, [10.0, 30.0])
        # Let assert_copy test that y is a view or copy
        with suppress(TypeError, ValueError):
            y[:] = 40


def test_copy_invalid():
    a = np.asarray([1, 2, 3])
    with pytest.raises(ValueError, match="copy"):
        at(a, 0).set(4, copy="invalid")


def test_xp():
    a = np.asarray([1, 2, 3])
    b = at(a, 0).set(4, xp=np)
    assert_array_equal(b, [4, 2, 3])
