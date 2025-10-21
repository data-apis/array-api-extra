from types import ModuleType

import numpy as np
from scipy.stats._axis_nan_policy import _broadcast_arrays

from ._at import at
from ._utils._compat import device as get_device
from ._utils._helpers import eager_shape
from ._utils._typing import Array


def quantile(  # numpydoc ignore=PR01,RT01
    x,
    p,
    /,
    method: str = 'linear',  # noqa: ARG001
    axis: int | None = None,
    keepdims: bool = False,
    *,
    xp: ModuleType,
):
    """See docstring in `array_api_extra._delegation.py`."""
    # Input validation / standardization
    temp = _quantile_iv(x, p, axis, keepdims)
    y, p, axis, keepdims, n, axis_none, ndim = temp

    res = _quantile_hf(y, p, n, xp)

    # Reshape per axis/keepdims
    if axis_none and keepdims:
        shape = (1,)*(ndim - 1) + res.shape
        res = xp.reshape(res, shape)
        axis = -1

    res = xp.moveaxis(res, -1, axis)

    if not keepdims:
        res = xp.squeeze(res, axis=axis)

    return res[()] if res.ndim == 0 else res


def _quantile_iv(
    x: Array,
    p: Array,
    axis: int | None,
    keepdims: bool,
    xp: ModuleType
):

    if not xp.isdtype(xp.asarray(x).dtype, ('integral', 'real floating')):
        raise ValueError("`x` must have real dtype.")

    if not xp.isdtype(xp.asarray(p).dtype, 'real floating'):
        raise ValueError("`p` must have real floating dtype.")

    p_mask = (p > 1) | (p < 0) | xp.isnan(p)
    if xp.any(p_mask):
        raise ValueError("`p` values must be in the range [0, 1]")

    device = get_device(x)
    floating_dtype = xp.result_type(x, p)
    x = xp.asarray(x, dtype=floating_dtype, device=device)
    p = xp.asarray(p, dtype=floating_dtype, device=device)
    dtype = x.dtype

    axis_none = axis is None
    ndim = max(x.ndim, p.ndim)
    if axis_none:
        x = xp.reshape(x, (-1,))
        p = xp.reshape(p, (-1,))
        axis = 0
    elif np.iterable(axis) or int(axis) != axis:
        message = "`axis` must be an integer or None."
        raise ValueError(message)
    elif (axis >= ndim) or (axis < -ndim):
        message = "`axis` is not compatible with the shapes of the inputs."
        raise ValueError(message)
    axis = int(axis)

    if keepdims not in {None, True, False}:
        message = "If specified, `keepdims` must be True or False."
        raise ValueError(message)

    # If data has length zero along `axis`, the result will be an array of NaNs just
    # as if the data had length 1 along axis and were filled with NaNs.
    n = eager_shape(x, axis)
    if n == 0:
        shape = eager_shape(x)
        shape[axis] = 1
        n = 1
        x = xp.full(shape, xp.nan, dtype=dtype, device=device)

    y = xp.sort(x, axis=axis, stable=False)
    # FIXME: I still need to look into the broadcasting:
    y, p = _broadcast_arrays((y, p), axis=axis)

    p_shape = eager_shape(p)
    if (keepdims is False) and (p_shape[axis] != 1):
        message = "`keepdims` may be False only if the length of `p` along `axis` is 1."
        raise ValueError(message)
    keepdims = (p_shape[axis] != 1) if keepdims is None else keepdims

    y = xp.moveaxis(y, axis, -1)
    p = xp.moveaxis(p, axis, -1)

    nans = xp.isnan(y)
    nan_out = xp.any(nans, axis=-1)
    if xp.any(nan_out):
        y = xp.asarray(y, copy=True)  # ensure writable
        y = at(y, nan_out).set(xp.nan)

    return y, p, axis, keepdims, n, axis_none, ndim, xp


def _quantile_hf(y, p, n, xp):
    m = 1 - p
    jg = p*n + m - 1
    j = jg // 1
    g = jg % 1
    g[j < 0] = 0
    j = xp.clip(j, 0., n - 1)
    jp1 = xp.clip(j + 1, 0., n - 1)

    return ((1 - g) * xp.take_along_axis(y, xp.astype(j, xp.int64), axis=-1)
            + g * xp.take_along_axis(y, xp.astype(jp1, xp.int64), axis=-1))
