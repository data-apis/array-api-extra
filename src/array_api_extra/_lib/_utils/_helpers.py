"""Helper functions used by `array_api_extra/_funcs.py`."""

# https://github.com/scikit-learn/scikit-learn/pull/27910#issuecomment-2568023972
from __future__ import annotations

from types import ModuleType
from typing import cast

from . import _compat
from ._compat import is_array_api_obj, is_numpy_array
from ._typing import Array

__all__ = ["mean"]


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
