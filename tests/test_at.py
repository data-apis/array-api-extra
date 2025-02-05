import math
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
from array_api_extra._lib._testing import xfail, xp_assert_equal
from array_api_extra._lib._utils._compat import array_namespace, is_writeable_array
from array_api_extra._lib._utils._typing import Array, Index
from array_api_extra.testing import lazy_xp_function

pytestmark = [
    pytest.mark.skip_xp_backend(
        Backend.SPARSE, reason="read-only backend without .at support"
    )
]


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


@pytest.mark.parametrize(
    ("kwargs", "expect_copy"),
    [
        pytest.param({"copy": True}, True, id="copy=True"),
        pytest.param({"copy": False}, False, id="copy=False"),
        # Behavior is backend-specific
        pytest.param({"copy": None}, None, id="copy=None"),
        # Test that the copy parameter defaults to None
        pytest.param({}, None, id="no copy kwarg"),
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
        (True, False),  # Uses xp.where(idx, y, x) on JAX and Dask
        pytest.param(
            True,
            True,
            marks=(
                pytest.mark.skip_xp_backend(  # test passes when copy=False
                    Backend.JAX, reason="bool mask update with shaped rhs"
                ),
                pytest.mark.xfail_xp_backend(
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


@pytest.mark.parametrize("copy", [True, None])
@pytest.mark.parametrize("bool_mask", [False, True])
@pytest.mark.parametrize("op", list(_AtOp))
def test_incompatible_dtype(
    xp: ModuleType,
    library: Backend,
    request: pytest.FixtureRequest,
    op: _AtOp,
    copy: bool | None,
    bool_mask: bool,
):
    """Test that at() replicates the backend's behaviour for
    in-place operations with incompatible dtypes.

    Behavior is backend-specific, but only two behaviors are allowed:
    1. raise an exception, or
    2. return the same dtype as x, disregarding y.dtype (no broadcasting).

    Note that __i<op>__ and __<op>__ behave differently, and we want to
    replicate the behavior of __i<op>__:

    >>> a = np.asarray([1, 2, 3])
    >>> a / 1.5
    array([0.        , 0.66666667, 1.33333333])
    >>> a /= 1.5
    UFuncTypeError: Cannot cast ufunc 'divide' output from dtype('float64')
    to dtype('int64') with casting rule 'same_kind'
    """
    x = xp.asarray([2, 4])
    idx = xp.asarray([True, False]) if bool_mask else slice(None)
    z = None

    if library is Backend.JAX:
        if bool_mask:
            z = at_op(x, idx, op, 1.1, copy=copy)
        else:
            with pytest.warns(FutureWarning, match="cannot safely cast"):
                z = at_op(x, idx, op, 1.1, copy=copy)

    elif library is Backend.DASK:
        if op in (_AtOp.MIN, _AtOp.MAX) and bool_mask:
            xfail(request, reason="need array-api-compat 1.11")
        z = at_op(x, idx, op, 1.1, copy=copy)

    elif library is Backend.ARRAY_API_STRICT and op is not _AtOp.SET:
        with pytest.raises(Exception, match=r"cast|promote|dtype"):
            at_op(x, idx, op, 1.1, copy=copy)

    elif op in (_AtOp.SET, _AtOp.MIN, _AtOp.MAX):
        # There is no __i<op>__ version of these operations
        z = at_op(x, idx, op, 1.1, copy=copy)

    else:
        with pytest.raises(Exception, match=r"cast|promote|dtype"):
            at_op(x, idx, op, 1.1, copy=copy)

    assert z is None or z.dtype == x.dtype


def test_bool_mask_nd(xp: ModuleType):
    x = xp.asarray([[1, 2, 3], [4, 5, 6]])
    idx = xp.asarray([[True, False, False], [False, True, True]])
    z = at_op(x, idx, _AtOp.SET, 0)
    xp_assert_equal(z, xp.asarray([[0, 2, 3], [4, 0, 0]]))


@pytest.mark.parametrize(
    "bool_mask",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail_xp_backend(
                Backend.DASK, reason="FIXME need scipy's lazywhere"
            ),
        ),
    ],
)
def test_no_inf_warnings(xp: ModuleType, bool_mask: bool):
    x = xp.asarray([math.inf, 1.0, 2.0])
    idx = ~xp.isinf(x) if bool_mask else slice(1, None)
    # inf - inf -> nan with a warning
    z = at_op(x, idx, _AtOp.SUBTRACT, math.inf)
    xp_assert_equal(z, xp.asarray([math.inf, -math.inf, -math.inf]))
