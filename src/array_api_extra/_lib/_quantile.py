"""Implementations of the quantile function."""

from types import ModuleType

from ._utils._compat import device as get_device
from ._utils._helpers import eager_shape
from ._utils._typing import Array


def quantile(  # numpydoc ignore=PR01,RT01
    a: Array,
    q: Array | float,
    /,
    method: str = "linear",  # noqa: ARG001
    axis: int | None = None,
    keepdims: bool = False,
    *,
    xp: ModuleType,
) -> Array:
    """See docstring in `array_api_extra._delegation.py`."""
    device = get_device(a)
    floating_dtype = xp.float64  # xp.result_type(a, xp.asarray(q))
    a = xp.asarray(a, dtype=floating_dtype, device=device)
    a_shape = list(a.shape)
    p: Array = xp.asarray(q, dtype=floating_dtype, device=device)

    q_scalar = p.ndim == 0
    if q_scalar:
        p = xp.reshape(p, (1,))

    axis_none = axis is None
    a_ndim = a.ndim
    if axis is None:
        a = xp.reshape(a, (-1,))
        axis = 0
    else:
        axis = int(axis)

    (n,) = eager_shape(a, axis)
    # If data has length zero along `axis`, the result will be an array of NaNs just
    # as if the data had length 1 along axis and were filled with NaNs.
    if n == 0:
        a_shape[axis] = 1
        n = 1
        a = xp.full(tuple(a_shape), xp.nan, dtype=floating_dtype, device=device)

    a = xp.sort(a, axis=axis, stable=False)
    # to support weights, the main thing would be to
    # argsort a, and then use it to sort a and w.
    # The hard part will be dealing with 0-weights and NaNs
    # But maybe a proper use of searchsorted + left/right side will work?

    res = _quantile_hf(a, p, float(n), axis, xp)

    # reshaping to conform to doc/other libs' behavior
    if axis_none:
        if keepdims:
            res = xp.reshape(res, p.shape + (1,) * a_ndim)
    else:
        res = xp.moveaxis(res, axis, 0)
        if keepdims:
            a_shape[axis] = 1
            res = xp.reshape(res, p.shape + tuple(a_shape))

    return res[0, ...] if q_scalar else res


def _quantile_hf(  # numpydoc ignore=GL08
    a: Array, q: Array, n: float, axis: int, xp: ModuleType
) -> Array:
    m = 1 - q
    jg = q * n + m - 1

    j = jg // 1
    j = xp.clip(j, 0.0, n - 1)
    jp1 = xp.clip(j + 1, 0.0, n - 1)
    # `̀j` and `jp1` are 1d arrays

    g = jg % 1
    g = xp.where(j < 0, 0, g)  # equivalent to g[j < 0] = 0, but works with strictest
    new_g_shape = [1] * a.ndim
    new_g_shape[axis] = g.shape[0]
    g = xp.reshape(g, tuple(new_g_shape))

    return (1 - g) * xp.take(a, xp.astype(j, xp.int64), axis=axis) + g * xp.take(
        a, xp.astype(jp1, xp.int64), axis=axis
    )
