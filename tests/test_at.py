from collections.abc import Callable, Generator
from contextlib import contextmanager
from enum import Enum
from importlib import import_module
from typing import cast

import numpy as np
import pytest
from array_api_compat import (  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
    array_namespace,
    is_dask_array,
    is_pydata_sparse_array,
    is_writeable_array,
)

from array_api_extra import at
from array_api_extra._funcs import _AtOp
from array_api_extra._lib._typing import Array


class Library(Enum):
    ARRAY_API_STRICT = "array_api_strict"
    NUMPY = "numpy"
    NUMPY_READONLY = "numpy_readonly"
    CUPY = "cupy"
    TORCH = "torch"
    DASK_ARRAY = "dask.array"
    SPARSE = "sparse"
    JAX_NUMPY = "jax.numpy"

    # @override from Python 3.12
    def __str__(self) -> str:  # type: ignore[explicit-override]  # pyright: ignore[reportImplicitOverride]
        return self.value


@pytest.fixture(params=tuple(Library))
def array(request: pytest.FixtureRequest) -> Array:
    library = request.param
    if library is Library.NUMPY_READONLY:
        x = np.asarray([10.0, 20.0, 30.0])
        x.flags.writeable = False
    else:
        xp = pytest.importorskip(library.value)
        x = xp.asarray([10.0, 20.0, 30.0])
    return x


def assert_array_equal(a: Array, b: Array) -> None:
    xp = array_namespace(a)
    b = xp.asarray(b)
    eq = xp.all(a == b)
    if is_dask_array(a):
        eq = eq.compute()
    assert eq


@contextmanager
def assert_copy(array: Array, copy: bool | None) -> Generator[None, None, None]:
    if copy is False and not is_writeable_array(array):
        with pytest.raises((TypeError, ValueError)):
            yield
        return

    xp = array_namespace(array)
    array_orig = xp.asarray(array, copy=True)
    yield

    if copy is None:
        copy = not is_writeable_array(array)
    assert_array_equal(xp.all(array == array_orig), copy)


@pytest.mark.parametrize(
    ("kwargs", "expect_copy"),
    [
        ({"copy": True}, True),
        ({"copy": False}, False),
        ({"copy": None}, None),  # Behavior is backend-specific
        ({}, None),  # Test that the copy parameter defaults to None
    ],
)
@pytest.mark.parametrize(
    ("op", "arg", "expect"),
    [
        (_AtOp.SET, 40.0, [10.0, 40.0, 40.0]),
        (_AtOp.ADD, 40.0, [10.0, 60.0, 70.0]),
        (_AtOp.SUBTRACT, 100.0, [10.0, -80.0, -70.0]),
        (_AtOp.MULTIPLY, 2.0, [10.0, 40.0, 60.0]),
        (_AtOp.DIVIDE, 2.0, [10.0, 10.0, 15.0]),
        (_AtOp.POWER, 2.0, [10.0, 400.0, 900.0]),
        (_AtOp.MIN, 25.0, [10.0, 20.0, 25.0]),
        (_AtOp.MAX, 25.0, [10.0, 25.0, 30.0]),
    ],
)
def test_update_ops(
    array: Array,
    kwargs: dict[str, bool | None],
    expect_copy: bool | None,
    op: _AtOp,
    arg: float,
    expect: list[float],
):
    if is_pydata_sparse_array(array):
        pytest.skip("at() does not support updates on sparse arrays")

    with assert_copy(array, expect_copy):
        func = cast(Callable[..., Array], getattr(at(array)[1:], op.value))  # type: ignore[no-any-explicit]
        y = func(arg, **kwargs)
        assert isinstance(y, type(array))
        assert_array_equal(y, expect)


def test_copy_invalid():
    a = np.asarray([1, 2, 3])
    with pytest.raises(ValueError, match="copy"):
        at(a, 0).set(4, copy="invalid")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def test_xp():
    a = np.asarray([1, 2, 3])
    at(a, 0).set(4, xp=np)
    at(a, 0).add(4, xp=np)
    at(a, 0).subtract(4, xp=np)
    at(a, 0).multiply(4, xp=np)
    at(a, 0).divide(4, xp=np)
    at(a, 0).power(4, xp=np)
    at(a, 0).min(4, xp=np)
    at(a, 0).max(4, xp=np)


def test_alternate_index_syntax():
    a = np.asarray([1, 2, 3])
    assert_array_equal(at(a, 0).set(4, copy=True), [4, 2, 3])
    assert_array_equal(at(a)[0].set(4, copy=True), [4, 2, 3])

    a_at = at(a)
    assert_array_equal(a_at[0].add(1, copy=True), [2, 2, 3])
    assert_array_equal(a_at[1].add(2, copy=True), [1, 4, 3])

    with pytest.raises(ValueError, match="Index"):
        at(a).set(4)
    with pytest.raises(ValueError, match="Index"):
        at(a, 0)[0].set(4)


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize(
    "op", [_AtOp.ADD, _AtOp.SUBTRACT, _AtOp.MULTIPLY, _AtOp.DIVIDE, _AtOp.POWER]
)
def test_iops_incompatible_dtype(op: _AtOp, copy: bool):
    """Test that at() replicates the backend's behaviour for
    in-place operations with incompatible dtypes.

    Note:
    >>> a = np.asarray([1, 2, 3])
    >>> a / 1.5
    array([0.        , 0.66666667, 1.33333333])
    >>> a /= 1.5
    UFuncTypeError: Cannot cast ufunc 'divide' output from dtype('float64')
    to dtype('int64') with casting rule 'same_kind'
    """
    a = np.asarray([2, 4])
    func = cast(Callable[..., Array], getattr(at(a)[:], op.value))  # type: ignore[no-any-explicit]
    with pytest.raises(TypeError, match="Cannot cast ufunc"):
        func(1.1, copy=copy)
