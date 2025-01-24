import pickle
from collections.abc import Callable, Generator
from contextlib import contextmanager
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest

from array_api_extra import at
from array_api_extra._lib import Backend
from array_api_extra._lib._at import _AtOp
from array_api_extra._lib._testing import xp_assert_equal
from array_api_extra._lib._utils._compat import array_namespace, is_writeable_array
from array_api_extra._lib._utils._typing import Array, Index
from array_api_extra.testing import lazy_xp_function


def at_op(  # type: ignore[no-any-explicit]
    x: Array,
    idx: Index,
    op: _AtOp,
    y: Array | object,
    **kwargs: Any,  # Test the default copy=None
) -> Array:
    """
    Wrapper around at(x, idx).op(y, copy=copy, xp=xp).

    This is a hack to allow wrapping `at()` with `lazy_xp_function`.
    For clarity, at() itself works inside jax.jit without hacks; this is
    just a workaround for when one wants to apply jax.jit to `at()` directly,
    which is not a common use case.
    """
    if isinstance(idx, (slice | tuple)):
        return _at_op(x, None, pickle.dumps(idx), op, y, **kwargs)
    return _at_op(x, idx, None, op, y, **kwargs)


def _at_op(  # type: ignore[no-any-explicit]
    x: Array,
    idx: Index | None,
    idx_pickle: bytes | None,
    op: _AtOp,
    y: Array | object,
    **kwargs: Any,
) -> Array:
    """jitted helper of at_op"""
    if idx_pickle:
        idx = pickle.loads(idx_pickle)
    meth = cast(Callable[..., Array], getattr(at(x, idx), op.value))  # type: ignore[no-any-explicit]
    return meth(y, **kwargs)


lazy_xp_function(_at_op, static_argnames=("op", "idx_pickle", "copy", "xp"))


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
    xp_assert_equal(xp.all(array == array_orig), xp.asarray(copy))


@pytest.mark.skip_xp_backend(
    Backend.SPARSE, reason="read-only backend without .at support"
)
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
    ("op", "y", "expect"),
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
@pytest.mark.parametrize(
    ("bool_mask", "shaped_y"),
    [
        (False, False),
        (False, True),
        pytest.param(
            True,
            False,
            marks=(
                pytest.mark.skip_xp_backend(Backend.JAX, reason="TODO special case"),
                pytest.mark.skip_xp_backend(Backend.DASK, reason="TODO special case"),
            ),
        ),
        pytest.param(
            True,
            True,
            marks=(
                pytest.mark.skip_xp_backend(
                    Backend.JAX, reason="bool mask update with shaped rhs"
                ),
                pytest.mark.skip_xp_backend(
                    Backend.DASK, reason="bool mask update with shaped rhs"
                ),
            ),
        ),
    ],
)
def test_update_ops(
    xp: ModuleType,
    kwargs: dict[str, bool | None],
    expect_copy: bool | None,
    op: _AtOp,
    y: float,
    expect: list[float],
    bool_mask: bool,
    shaped_y: bool,
):
    x = xp.asarray([10.0, 20.0, 30.0])
    idx = xp.asarray([False, True, True]) if bool_mask else slice(1, None)
    if shaped_y:
        y = xp.asarray([y, y])

    with assert_copy(x, expect_copy):
        z = at_op(x, idx, op, y, **kwargs)
        assert isinstance(z, type(x))
        xp_assert_equal(z, xp.asarray(expect))


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
    xp_assert_equal(at(a, 0).set(4, copy=True), np.asarray([4, 2, 3]))
    xp_assert_equal(at(a)[0].set(4, copy=True), np.asarray([4, 2, 3]))

    a_at = at(a)
    xp_assert_equal(a_at[0].add(1, copy=True), np.asarray([2, 2, 3]))
    xp_assert_equal(a_at[1].add(2, copy=True), np.asarray([1, 4, 3]))

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
    x = np.asarray([2, 4])
    with pytest.raises(TypeError, match="Cannot cast ufunc"):
        at_op(x, slice(None), op, 1.1, copy=copy)
