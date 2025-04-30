import math
from collections.abc import Callable, Generator
from contextlib import contextmanager
from types import ModuleType
from typing import cast

import numpy as np
import pytest

from array_api_extra import at
from array_api_extra._lib._at import _AtOp
from array_api_extra._lib._backends import Backend
from array_api_extra._lib._testing import xp_assert_equal
from array_api_extra._lib._utils._compat import array_namespace, is_writeable_array
from array_api_extra._lib._utils._compat import device as get_device
from array_api_extra._lib._utils._typing import Array, Device, SetIndex
from array_api_extra.testing import lazy_xp_function

pytestmark = [
    pytest.mark.skip_xp_backend(
        Backend.SPARSE, reason="read-only backend without .at support"
    ),
    pytest.mark.skip_xp_backend(Backend.ARRAY_API_STRICTEST, reason="boolean indexing"),
]


def at_op(
    x: Array,
    idx: SetIndex,
    op: _AtOp,
    y: Array | object,
    copy: bool | None = None,
    xp: ModuleType | None = None,
) -> Array:
    """
    Wrapper around at(x, idx).op(y, copy=copy, xp=xp).

    This is a hack to allow wrapping `at()` with `lazy_xp_function`.
    For clarity, at() itself works inside jax.jit without hacks; this is
    just a workaround for when one wants to apply jax.jit to `at()` directly,
    which is not a common use case.
    """
    meth = cast(Callable[..., Array], getattr(at(x, idx), op.value))  # type: ignore[explicit-any]
    return meth(y, copy=copy, xp=xp)


lazy_xp_function(at_op)


@contextmanager
def assert_copy(
    array: Array, copy: bool | None, expect_copy: bool | None = None
) -> Generator[None, None, None]:
    if copy is False and not is_writeable_array(array):
        with pytest.raises((TypeError, ValueError)):
            yield
        return

    xp = array_namespace(array)
    array_orig = xp.asarray(array, copy=True)
    yield

    if expect_copy is None:
        expect_copy = copy

    if expect_copy:
        # Original has not been modified
        xp_assert_equal(array, array_orig)
    elif expect_copy is False:
        # Original has been modified
        with pytest.raises(AssertionError):
            xp_assert_equal(array, array_orig)
    # Test nothing for copy=None. Dask changes behaviour depending on
    # whether it's a special case of a bool mask with scalar RHS or not.


@pytest.mark.parametrize("copy", [False, True, None])
@pytest.mark.parametrize(
    ("op", "y", "expect_list"),
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
    ("bool_mask", "x_ndim", "y_ndim"),
    [
        (False, 1, 0),
        (False, 1, 1),
        (True, 1, 0),  # Uses xp.where(idx, y, x) on JAX and Dask
        pytest.param(
            *(True, 1, 1),
            marks=(
                pytest.mark.xfail_xp_backend(
                    Backend.JAX,
                    reason="bool mask update with shaped rhs",
                    strict=False,  # test passes when copy=False
                ),
                pytest.mark.xfail_xp_backend(
                    Backend.JAX_GPU,
                    reason="bool mask update with shaped rhs",
                    strict=False,  # test passes when copy=False
                ),
                pytest.mark.xfail_xp_backend(
                    Backend.DASK, reason="bool mask update with shaped rhs"
                ),
            ),
        ),
        (False, 0, 0),
        (True, 0, 0),
    ],
)
def test_update_ops(
    xp: ModuleType,
    copy: bool | None,
    op: _AtOp,
    y: float,
    expect_list: list[float],
    bool_mask: bool,
    x_ndim: int,
    y_ndim: int,
):
    if x_ndim == 1:
        x = xp.asarray([10.0, 20.0, 30.0])
        idx = xp.asarray([False, True, True]) if bool_mask else slice(1, None)
        expect: list[float] | float = expect_list
    else:
        idx = xp.asarray(True) if bool_mask else ()
        # Pick an element that does change with the operation
        if op is _AtOp.MIN:
            x = xp.asarray(30.0)
            expect = expect_list[2]
        else:
            x = xp.asarray(20.0)
            expect = expect_list[1]

    if y_ndim == 1:
        y = xp.asarray([y, y])

    with assert_copy(x, copy):
        z = at_op(x, idx, op, y, copy=copy)
        assert isinstance(z, type(x))
        xp_assert_equal(z, xp.asarray(expect))


@pytest.mark.parametrize("op", list(_AtOp))
def test_copy_default(xp: ModuleType, library: Backend, op: _AtOp):
    """
    Test that the default copy behaviour is False for writeable arrays
    and True for read-only ones.
    """
    x = xp.asarray([1.0, 10.0, 20.0])
    expect_copy = not is_writeable_array(x)
    meth = cast(Callable[..., Array], getattr(at(x)[:2], op.value))  # type: ignore[explicit-any]
    with assert_copy(x, None, expect_copy):
        _ = meth(2.0)

    x = xp.asarray([1.0, 10.0, 20.0])
    # Dask's default copy value is True for bool masks,
    # even if the arrays are writeable.
    expect_copy = not is_writeable_array(x) or library is Backend.DASK
    idx = xp.asarray([True, True, False])
    meth = cast(Callable[..., Array], getattr(at(x, idx), op.value))  # type: ignore[explicit-any]
    with assert_copy(x, None, expect_copy):
        _ = meth(2.0)


def test_copy_invalid():
    a = np.asarray([1, 2, 3])
    with pytest.raises(ValueError, match="copy"):
        _ = at(a, 0).set(4, copy="invalid")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def test_xp():
    a = cast(Array, np.asarray([1, 2, 3]))  # type: ignore[bad-cast]
    _ = at(a, 0).set(4, xp=np)
    _ = at(a, 0).add(4, xp=np)
    _ = at(a, 0).subtract(4, xp=np)
    _ = at(a, 0).multiply(4, xp=np)
    _ = at(a, 0).divide(4, xp=np)
    _ = at(a, 0).power(4, xp=np)
    _ = at(a, 0).min(4, xp=np)
    _ = at(a, 0).max(4, xp=np)


def test_alternate_index_syntax():
    xp = cast(ModuleType, np)  # pyright: ignore[reportInvalidCast]
    a = cast(Array, xp.asarray([1, 2, 3]))
    xp_assert_equal(at(a, 0).set(4, copy=True), xp.asarray([4, 2, 3]))
    xp_assert_equal(at(a)[0].set(4, copy=True), xp.asarray([4, 2, 3]))

    a_at = at(a)
    xp_assert_equal(a_at[0].add(1, copy=True), xp.asarray([2, 2, 3]))
    xp_assert_equal(a_at[1].add(2, copy=True), xp.asarray([1, 4, 3]))

    with pytest.raises(ValueError, match="Index"):
        _ = at(a).set(4)
    with pytest.raises(ValueError, match="Index"):
        _ = at(a, 0)[0].set(4)


@pytest.mark.parametrize("copy", [True, None])
@pytest.mark.parametrize("bool_mask", [False, True])
@pytest.mark.parametrize("op", list(_AtOp))
def test_incompatible_dtype(
    xp: ModuleType,
    library: Backend,
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

    if library.like(Backend.JAX):
        if bool_mask:
            z = at_op(x, idx, op, 1.1, copy=copy)
        else:
            with pytest.warns(FutureWarning, match="cannot safely cast"):
                z = at_op(x, idx, op, 1.1, copy=copy)

    elif library.like(Backend.DASK):
        z = at_op(x, idx, op, 1.1, copy=copy)

    elif library.like(Backend.ARRAY_API_STRICT) and op is not _AtOp.SET:
        with pytest.raises(Exception, match=r"cast|promote|dtype"):
            _ = at_op(x, idx, op, 1.1, copy=copy)

    elif op in (_AtOp.SET, _AtOp.MIN, _AtOp.MAX):
        # There is no __i<op>__ version of these operations
        z = at_op(x, idx, op, 1.1, copy=copy)

    else:
        with pytest.raises(Exception, match=r"cast|promote|dtype"):
            _ = at_op(x, idx, op, 1.1, copy=copy)

    assert z is None or z.dtype == x.dtype


def test_bool_mask_nd(xp: ModuleType):
    x = xp.asarray([[1, 2, 3], [4, 5, 6]])
    idx = xp.asarray([[True, False, False], [False, True, True]])
    z = at_op(x, idx, _AtOp.SET, 0)
    xp_assert_equal(z, xp.asarray([[0, 2, 3], [4, 0, 0]]))


@pytest.mark.parametrize("bool_mask", [False, True])
def test_no_inf_warnings(xp: ModuleType, bool_mask: bool):
    x = xp.asarray([math.inf, 1.0, 2.0])
    idx = ~xp.isinf(x) if bool_mask else slice(1, None)
    # inf - inf -> nan with a warning
    z = at_op(x, idx, _AtOp.SUBTRACT, math.inf)
    xp_assert_equal(z, xp.asarray([math.inf, -math.inf, -math.inf]))


@pytest.mark.parametrize(
    "copy",
    [
        None,
        pytest.param(
            False,
            marks=[
                pytest.mark.skip_xp_backend(
                    Backend.NUMPY, reason="np.generic is read-only"
                ),
                pytest.mark.skip_xp_backend(
                    Backend.NUMPY_READONLY, reason="read-only backend"
                ),
                pytest.mark.skip_xp_backend(Backend.JAX, reason="read-only backend"),
                pytest.mark.skip_xp_backend(
                    Backend.JAX_GPU, reason="read-only backend"
                ),
                pytest.mark.skip_xp_backend(Backend.SPARSE, reason="read-only backend"),
            ],
        ),
    ],
)
@pytest.mark.parametrize("bool_mask", [False, True])
def test_gh134(xp: ModuleType, bool_mask: bool, copy: bool | None):
    """
    Test that xpx.at doesn't encroach in a bug of dask.array.Array.__setitem__, which
    blindly assumes that chunk contents are writeable np.ndarray objects:

    https://github.com/dask/dask/issues/11722

    In other words: when special-casing bool masks for Dask, unless the user explicitly
    asks for copy=False, do not needlessly write back to the input.
    """
    x = xp.zeros(1)

    # In NumPy, we have a writeable np.ndarray in input and a read-only np.generic in
    # output. As both are Arrays, this behaviour is Array API compliant.
    # In Dask, we have a writeable da.Array on both sides, and if you call __setitem__
    # on it all seems fine, but when you compute() your graph is corrupted.
    y = x[0]

    idx = xp.asarray(True) if bool_mask else ()
    z = at_op(y, idx, _AtOp.SET, 1, copy=copy)
    xp_assert_equal(z, xp.asarray(1, dtype=x.dtype))


def test_device(xp: ModuleType, device: Device):
    x = xp.asarray([1, 2, 3], device=device)

    y = xp.asarray([4, 5], device=device)
    z = at(x)[:2].set(y)
    assert get_device(z) == get_device(x)

    idx = xp.asarray([True, False, True], device=device)
    z = at(x)[idx].set(4)
    assert get_device(z) == get_device(x)
