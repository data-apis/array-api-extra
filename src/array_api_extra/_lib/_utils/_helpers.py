"""Helper functions used by `array_api_extra/_funcs.py`."""

# https://github.com/scikit-learn/scikit-learn/pull/27910#issuecomment-2568023972
from __future__ import annotations

from types import ModuleType
from typing import cast

from . import _compat
from ._compat import is_array_api_obj, is_numpy_array
from ._typing import Array

__all__ = ["in1d", "mean"]


def in1d(
    x1: Array,
    x2: Array,
    /,
    *,
    assume_unique: bool = False,
    invert: bool = False,
    xp: ModuleType | None = None,
) -> Array:  # numpydoc ignore=PR01,RT01
    """
    Check whether each element of an array is also present in a second array.

    Returns a boolean array the same length as `x1` that is True
    where an element of `x1` is in `x2` and False otherwise.

    This function has been adapted using the original implementation
    present in numpy:
    https://github.com/numpy/numpy/blob/v1.26.0/numpy/lib/arraysetops.py#L524-L758
    """
    if xp is None:
        xp = _compat.array_namespace(x1, x2)

    # This code is run to make the code significantly faster
    if x2.shape[0] < 10 * x1.shape[0] ** 0.145:
        if invert:
            mask = xp.ones(x1.shape[0], dtype=xp.bool, device=_compat.device(x1))
            for a in x2:
                mask &= x1 != a
        else:
            mask = xp.zeros(x1.shape[0], dtype=xp.bool, device=_compat.device(x1))
            for a in x2:
                mask |= x1 == a
        return mask

    rev_idx = xp.empty(0)  # placeholder
    if not assume_unique:
        x1, rev_idx = xp.unique_inverse(x1)
        x2 = xp.unique_values(x2)

    ar = xp.concat((x1, x2))
    device_ = _compat.device(ar)
    # We need this to be a stable sort.
    order = xp.argsort(ar, stable=True)
    reverse_order = xp.argsort(order, stable=True)
    sar = xp.take(ar, order, axis=0)
    ar_size = _compat.size(sar)
    assert ar_size is not None, "xp.unique*() on lazy backends raises"
    if ar_size >= 1:
        bool_ar = sar[1:] != sar[:-1] if invert else sar[1:] == sar[:-1]
    else:
        bool_ar = xp.asarray([False]) if invert else xp.asarray([True])
    flag = xp.concat((bool_ar, xp.asarray([invert], device=device_)))
    ret = xp.take(flag, reverse_order, axis=0)

    if assume_unique:
        return ret[: x1.shape[0]]
    return xp.take(ret, rev_idx, axis=0)


def mean(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    xp: ModuleType | None = None,
) -> Array:  # numpydoc ignore=PR01,RT01
    """
    Complex mean, https://github.com/data-apis/array-api/issues/846.
    """
    if xp is None:
        xp = _compat.array_namespace(x)

    if xp.isdtype(x.dtype, "complex floating"):
        x_real = xp.real(x)
        x_imag = xp.imag(x)
        mean_real = xp.mean(x_real, axis=axis, keepdims=keepdims)
        mean_imag = xp.mean(x_imag, axis=axis, keepdims=keepdims)
        return mean_real + (mean_imag * xp.asarray(1j))
    return xp.mean(x, axis=axis, keepdims=keepdims)


def is_python_scalar(x: object) -> bool:  # numpydoc ignore=PR01,RT01
    """Return True if `x` is a Python scalar, False otherwise."""
    # isinstance(x, float) returns True for np.float64
    # isinstance(x, complex) returns True for np.complex128
    return isinstance(x, int | float | complex | bool) and not is_numpy_array(x)


def asarrays(
    a: Array | int | float | complex | bool,
    b: Array | int | float | complex | bool,
    xp: ModuleType,
) -> tuple[Array, Array]:
    """
    Ensure both `a` and `b` are arrays.

    If `b` is a python scalar, it is converted to the same dtype as `a`, and vice versa.

    Behavior is not specified when mixing a Python ``float`` and an array with an
    integer data type; this may give ``float32``, ``float64``, or raise an exception.
    Behavior is implementation-specific.

    Similarly, behavior is not specified when mixing a Python ``complex`` and an array
    with a real-valued data type; this may give ``complex64``, ``complex128``, or raise
    an exception. Behavior is implementation-specific.

    Parameters
    ----------
    a, b : Array | int | float | complex | bool
        Input arrays or scalars. At least one must be an array.
    xp : ModuleType
        The standard-compatible namespace for the returned arrays.

    Returns
    -------
    Array, Array
        The input arrays, possibly converted to arrays if they were scalars.

    See Also
    --------
    mixing-arrays-with-python-scalars : Array API specification for the behavior.
    """
    a_scalar = is_python_scalar(a)
    b_scalar = is_python_scalar(b)
    if not a_scalar and not b_scalar:
        return a, b  # This includes misc. malformed input e.g. str

    swap = False
    if a_scalar:
        swap = True
        b, a = a, b

    if is_array_api_obj(a):
        # a is an Array API object
        # b is a int | float | complex | bool

        # pyright doesn't like it if you reuse the same variable name
        xa = cast(Array, a)

        # https://data-apis.org/array-api/draft/API_specification/type_promotion.html#mixing-arrays-with-python-scalars
        same_dtype = {
            bool: "bool",
            int: ("integral", "real floating", "complex floating"),
            float: ("real floating", "complex floating"),
            complex: "complex floating",
        }
        kind = same_dtype[type(b)]  # type: ignore[index]
        if xp.isdtype(xa.dtype, kind):
            xb = xp.asarray(b, dtype=xa.dtype)
        else:
            # Undefined behaviour. Let the function deal with it, if it can.
            xb = xp.asarray(b)

    else:
        # Neither a nor b are Array API objects.
        # Note: we can only reach this point when one explicitly passes
        # xp=xp to the calling function; otherwise we fail earlier on
        # array_namespace(a, b).
        xa, xb = xp.asarray(a), xp.asarray(b)

    return (xb, xa) if swap else (xa, xb)
