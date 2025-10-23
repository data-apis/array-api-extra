"""Implementations of the quantile function."""

from types import ModuleType

from ._utils._compat import device as get_device
from ._utils._helpers import eager_shape
from ._utils._typing import Array, Device


def quantile(  # numpydoc ignore=PR01,RT01
    a: Array,
    q: Array,
    /,
    method: str = "linear",
    axis: int | None = None,
    keepdims: bool = False,
    nan_policy: str = "propagate",
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
        res = _quantile(a, q, n, axis, method, xp)
        if not axis_none:
            res = xp.moveaxis(res, axis, 0)
    else:
        weights = xp.asarray(weights, dtype=xp.float64, device=device)
        average = method == 'averaged_inverted_cdf'
        res = _weighted_quantile(
            a, q, weights, n, axis, average,
            nan_policy=nan_policy, xp=xp, device=device
        )

    # reshaping to conform to doc/other libs' behavior
    if axis_none:
        if keepdims:
            res = xp.reshape(res, q.shape + (1,) * a_ndim)
    elif keepdims:
        a_shape[axis] = 1
        res = xp.reshape(res, q.shape + tuple(a_shape))

    return res[0, ...] if q_scalar else res


def _quantile(  # numpydoc ignore=GL08
    a: Array, q: Array, n: int, axis: int, method: str, xp: ModuleType
) -> Array:
    a = xp.sort(a, axis=axis, stable=False)
    mask_nan = xp.any(xp.isnan(a), axis=axis, keepdims=True)
    if xp.any(mask_nan):
        # propogate NaNs:
        mask = xp.repeat(mask_nan, n, axis=axis)
        a = xp.where(mask, xp.nan, a)
        del mask

    if method == "linear":
        m = 1 - q
    else: # method is "inverted_cdf" or "averaged_inverted_cdf"
        m = 0

    jg = q * float(n) + m - 1

    j = jg // 1
    j = xp.clip(j, 0.0, float(n - 1))
    jp1 = xp.clip(j + 1, 0.0, float(n - 1))
    # `̀j` and `jp1` are 1d arrays

    g = jg % 1
    if method == 'inverted_cdf':
        g = xp.astype((g > 0), jg.dtype)
    elif method == 'averaged_inverted_cdf':
        g = (1 + xp.astype((g > 0), jg.dtype)) / 2

    g = xp.where(j < 0, 0, g)  # equivalent to g[j < 0] = 0, but works with readonly
    new_g_shape = [1] * a.ndim
    new_g_shape[axis] = g.shape[0]
    g = xp.reshape(g, tuple(new_g_shape))

    return (1 - g) * xp.take(a, xp.astype(j, xp.int64), axis=axis) + g * xp.take(
        a, xp.astype(jp1, xp.int64), axis=axis
    )


def _weighted_quantile(
    a: Array, q: Array, weights: Array, n: int, axis: int, average: bool, nan_policy: str,
    xp: ModuleType, device: Device
) -> Array:
    """
    a is expected to be 1d or 2d.
    """
    kwargs = dict(n=n, average=average, nan_policy=nan_policy, xp=xp, device=device)
    a = xp.moveaxis(a, axis, -1)
    if weights.ndim > 1:
        weights = xp.moveaxis(weights, axis, -1)
    sorter = xp.argsort(a, axis=-1, stable=False)

    if a.ndim == 1:
        x = xp.take(a, sorter)
        w = xp.take(weights, sorter)
        return _weighted_quantile_sorted_1d(x, q, w, **kwargs)

    d, = eager_shape(a, axis=0)
    res = []
    for idx in range(d):
        w = weights if weights.ndim == 1 else weights[idx, ...]
        w = xp.take(w, sorter[idx, ...])
        x = xp.take(a[idx, ...], sorter[idx, ...])
        res.append(_weighted_quantile_sorted_1d(x, q, w, **kwargs))
    res = xp.stack(res, axis=1)
    return res


def _weighted_quantile_sorted_1d(
    x: Array, q: Array, w: Array, n: int, average: bool, nan_policy: str,
    xp: ModuleType, device: Device
) -> Array:
    if nan_policy == "omit":
        w = xp.where(xp.isnan(x), 0., w)
    elif xp.any(xp.isnan(x)):
        return xp.full(q.shape, xp.nan, dtype=x.dtype, device=device)
    cw = xp.cumulative_sum(w)
    t = cw[-1] * q
    i = xp.searchsorted(cw, t, side='left')
    j = xp.searchsorted(cw, t, side='right')
    i = xp.clip(i, 0, n - 1)
    j = xp.clip(j, 0, n - 1)

    # Ignore leading `weights=0` observations when `q=0`
    # see https://github.com/scikit-learn/scikit-learn/pull/20528
    i = xp.where(q == 0., j, i)
    if average:
        # Ignore trailing `weights=0` observations when `q=1`
        j = xp.where(q == 1., i, j)
        return (xp.take(x, i) + xp.take(x, j)) / 2
    return xp.take(x, i)
