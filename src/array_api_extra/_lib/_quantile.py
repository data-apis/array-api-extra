from types import ModuleType

import numpy as np
from scipy.stats._axis_nan_policy import _broadcast_arrays

from ._at import at
from ._utils._compat import device as get_device
from ._utils._helpers import eager_shape
from ._utils._typing import Array


def quantile(  # numpydoc ignore=PR01,RT01
    a: Array,
    q: Array | float,
    /,
    method: str = 'linear',  # noqa: ARG001
    axis: int | None = None,
    keepdims: bool = False,
    *,
    xp: ModuleType,
):
    """See docstring in `array_api_extra._delegation.py`."""
    device = get_device(a)
    floating_dtype = xp.result_type(a, xp.asarray(q))
    a = xp.asarray(a, dtype=floating_dtype, device=device)
    q = xp.asarray(q, dtype=floating_dtype, device=device)

    if xp.any((q > 1) | (q < 0) | xp.isnan(q)):
        raise ValueError("`q` values must be in the range [0, 1]")

    q_scalar = q.ndim == 0
    if q_scalar:
        q = xp.reshape(q, (1,))

    axis_none = axis is None
    if axis_none:
        a = xp.reshape(a, (-1,))
        axis = 0
    axis = int(axis)

    n = eager_shape(a, axis)
    # If data has length zero along `axis`, the result will be an array of NaNs just
    # as if the data had length 1 along axis and were filled with NaNs.
    if n == 0:
        shape = list(eager_shape(a))
        shape[axis] = 1
        n = 1
        a = xp.full(shape, xp.nan, dtype=floating_dtype, device=device)

    a = xp.sort(a, axis=axis, stable=False)
    # to support weights, the main thing would be to
    # argsort a, and then use it to sort a and w.
    # The hard part will be dealing with 0-weights and NaNs
    # But maybe a proper use of searchsorted + left/right side will work?

    res = _quantile_hf(a, q, n, axis, xp)

    # reshaping to conform to doc/other libs' behavior
    if axis_none:
        if keepdims:
            res = xp.reshape(res, q.shape + (1,) * a.ndim)
    else:
        res = xp.moveaxis(res, axis, 0)
        if keepdims:
            shape = list(a.shape)
            shape[axis] = 1
            shape = q.shape + tuple(shape)
            res = xp.reshape(res, shape)

    return res[0, ...] if q_scalar else res


def _quantile_hf(y: Array, p: Array, n: int, axis: int, xp: ModuleType):
    m = 1 - p
    jg = p*n + m - 1
    j = jg // 1
    g = jg % 1
    g[j < 0] = 0
    j = xp.clip(j, 0., n - 1)
    jp1 = xp.clip(j + 1, 0., n - 1)
    # `̀j` and `jp1` are 1d arrays

    return (
        (1 - g) * xp.take(y, xp.astype(j, xp.int64), axis=axis)
        + g * xp.take(y, xp.astype(jp1, xp.int64), axis=axis)
    )
