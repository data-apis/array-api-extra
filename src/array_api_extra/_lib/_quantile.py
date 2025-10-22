"""Implementations of the quantile function."""

from types import ModuleType

from ._utils._compat import device as get_device
from ._utils._helpers import eager_shape
from ._utils._typing import Array


def quantile(  # numpydoc ignore=PR01,RT01
    a: Array,
    q: Array,
    /,
    method: str = "linear",
    axis: int | None = None,
    keepdims: bool = False,
    *,
    weights: Array | None = None,
    xp: ModuleType,
) -> Array:
    """See docstring in `array_api_extra._delegation.py`."""
    device = get_device(a)
    a_shape = list(a.shape)

    q_scalar = q.ndim == 0
    if q_scalar:
        q = xp.reshape(q, (1,))

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
        a = xp.full(tuple(a_shape), xp.nan, dtype=a.dtype, device=device)

    if weights is None:
        res = _quantile(a, q, float(n), axis, method, xp)
    else:
        average = method == 'averaged_inverted_cdf'
        res = _weighted_quantile(a, q, weights, n, axis, average, xp)
    # to support weights, the main thing would be to
    # argsort a, and then use it to sort a and w.
    # The hard part will be dealing with 0-weights and NaNs
    # But maybe a proper use of searchsorted + left/right side will work?

    # reshaping to conform to doc/other libs' behavior
    if axis_none:
        if keepdims:
            res = xp.reshape(res, q.shape + (1,) * a_ndim)
    else:
        res = xp.moveaxis(res, axis, 0)
        if keepdims:
            a_shape[axis] = 1
            res = xp.reshape(res, q.shape + tuple(a_shape))

    return res[0, ...] if q_scalar else res


def _quantile(  # numpydoc ignore=GL08
    a: Array, q: Array, n: float, axis: int, method: str, xp: ModuleType
) -> Array:
    a = xp.sort(a, axis=axis, stable=False)

    if method == "linear":
        m = 1 - q       
    else: # method is "inverted_cdf" or "averaged_inverted_cdf"
        m = 0

    jg = q * n + m - 1

    j = jg // 1
    j = xp.clip(j, 0.0, n - 1)
    jp1 = xp.clip(j + 1, 0.0, n - 1)
    # `̀j` and `jp1` are 1d arrays

    g = jg % 1
    if method == 'inverted_cdf':
        g = xp.astype((g > 0), jg.dtype)
    elif method == 'averaged_inverted_cdf':
        g = (1 + xp.astype((g > 0), jg.dtype)) / 2

    g = xp.where(j < 0, 0, g)  # equivalent to g[j < 0] = 0, but works with strictest
    new_g_shape = [1] * a.ndim
    new_g_shape[axis] = g.shape[0]
    g = xp.reshape(g, tuple(new_g_shape))

    return (1 - g) * xp.take(a, xp.astype(j, xp.int64), axis=axis) + g * xp.take(
        a, xp.astype(jp1, xp.int64), axis=axis
    )


def _weighted_quantile(a: Array, q: Array, weights: Array, n: int, axis, average: bool, xp: ModuleType):
    a = xp.moveaxis(a, axis, -1)
    sorter = xp.argsort(a, axis=-1, stable=False)
    a = xp.take_along_axis(a, sorter, axis=-1)

    if a.ndim == 1:
        return _weighted_quantile_sorted_1d(a, q, weights, n, )

    d, = eager_shape(a, axis=0)
    res = xp.empty((q.shape[0], d))
    for idx in range(d):
        w = weights if weights.ndim == 1 else weights[idx, ...]
        w = xp.take(w, sorter[idx, ...])
        res[..., idx] = _weighted_quantile_sorted_1d(a[idx, ...], q, w, n, average)
    return res


def _weighted_quantile_sorted_1d(a, q, w, n, average: bool, xp: ModuleType):
    cw = xp.cumsum(w)
    t = cw[-1] * q
    i = xp.searchsorted(cw, t)
    j = xp.searchsorted(cw, t, side='right')
    i = xp.minimum(i, float(n - 1))
    j = xp.minimum(j, float(n - 1))

    # Ignore leading `weights=0` observations when `q=0`
    # see https://github.com/scikit-learn/scikit-learn/pull/20528
    i = xp.where(q == 0., j, i)   
    if average:
        # Ignore trailing `weights=0` observations when `q=1`
        j = xp.where(q == 1., i, j)
        return (xp.take(a, i) + xp.take(a, j)) / 2
    else:
        return xp.take(a, i)
