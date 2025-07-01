"""Quantile implementation."""

from types import ModuleType
from typing import cast

from ._at import at
from ._utils import _compat
from ._utils._compat import array_namespace
from ._utils._typing import Array


def quantile(
    x: Array,
    q: Array | float,
    /,
    *,
    axis: int | None = None,
    keepdims: bool | None = None,
    method: str = "linear",
    xp: ModuleType | None = None,
) -> Array:  # numpydoc ignore=PR01,RT01
    """See docstring in `array_api_extra._delegation.py`."""
    if xp is None:
        xp = array_namespace(x, q)

    q_is_scalar = isinstance(q, int | float)
    if q_is_scalar:
        q = xp.asarray(q, dtype=xp.float64, device=_compat.device(x))
    q_arr = cast(Array, q)

    if not xp.isdtype(x.dtype, ("integral", "real floating")):
        msg = "`x` must have real dtype."
        raise ValueError(msg)
    if not xp.isdtype(q_arr.dtype, "real floating"):
        msg = "`q` must have real floating dtype."
        raise ValueError(msg)

    # Promote to common dtype
    x = xp.astype(x, xp.float64)
    q_arr = xp.astype(q_arr, xp.float64)
    q_arr = xp.asarray(q_arr, device=_compat.device(x))

    dtype = x.dtype
    axis_none = axis is None
    ndim = max(x.ndim, q_arr.ndim)

    if axis_none:
        x = xp.reshape(x, (-1,))
        q_arr = xp.reshape(q_arr, (-1,))
        axis = 0
    elif not isinstance(axis, int):  # pyright: ignore[reportUnnecessaryIsInstance]
        msg = "`axis` must be an integer or None."
        raise ValueError(msg)
    elif axis >= ndim or axis < -ndim:
        msg = "`axis` is not compatible with the shapes of the inputs."
        raise ValueError(msg)
    else:
        axis = int(axis)

    if keepdims not in {None, True, False}:
        msg = "If specified, `keepdims` must be True or False."
        raise ValueError(msg)

    if x.shape[axis] == 0:
        shape = list(x.shape)
        shape[axis] = 1
        x = xp.full(shape, xp.nan, dtype=dtype, device=_compat.device(x))

    y = xp.sort(x, axis=axis)

    # Move axis to the end for easier processing
    y = xp.moveaxis(y, axis, -1)
    if not (q_is_scalar or q_arr.ndim == 0):
        q_arr = xp.moveaxis(q_arr, axis, -1)

    n = xp.asarray(y.shape[-1], dtype=dtype, device=_compat.device(y))

    # Validate that q values are in the range [0, 1]
    if xp.any((q_arr < 0) | (q_arr > 1)):
        msg = "`q` must contain values between 0 and 1 inclusive."
        raise ValueError(msg)

    res = _quantile_hf(y, q_arr, n, method, xp)

    # Reshape per axis/keepdims
    if axis_none and keepdims:
        shape = (1,) * (ndim - 1) + res.shape
        res = xp.reshape(res, shape)
        axis = -1

    # Move axis back to original position
    res = xp.moveaxis(res, -1, axis)

    # Handle keepdims
    if not keepdims and res.shape[axis] == 1:
        res = xp.squeeze(res, axis=axis)

    # For scalar q, ensure we return a scalar result
    # if q_is_scalar and hasattr(res, "shape") and res.shape != ():
    #    res = res[()]
    if res.ndim == 0:
        return res[()]
    return res


def _quantile_hf(
    y: Array, p: Array, n: Array, method: str, xp: ModuleType
) -> Array:  # numpydoc ignore=PR01,RT01
    """Helper function for Hyndman-Fan quantile method."""
    ms: dict[str, Array | int | float] = {
        "inverted_cdf": 0,
        "averaged_inverted_cdf": 0,
        "closest_observation": -0.5,
        "interpolated_inverted_cdf": 0,
        "hazen": 0.5,
        "weibull": p,
        "linear": 1 - p,
        "median_unbiased": p / 3 + 1 / 3,
        "normal_unbiased": p / 4 + 3 / 8,
    }
    m = ms[method]

    jg = p * n + m - 1
    # Convert both to integers, the type of j and n must be the same
    # for us to be able to `xp.clip` them.
    j = xp.astype(jg // 1, xp.int64)
    n = xp.astype(n, xp.int64)
    g = jg % 1

    if method == "inverted_cdf":
        g = xp.astype((g > 0), jg.dtype)
    elif method == "averaged_inverted_cdf":
        g = (1 + xp.astype((g > 0), jg.dtype)) / 2
    elif method == "closest_observation":
        g = 1 - xp.astype((g == 0) & (j % 2 == 1), jg.dtype)
    if method in {"inverted_cdf", "averaged_inverted_cdf", "closest_observation"}:
        g = xp.asarray(g)
        g = at(g, jg < 0).set(0)
        g = at(g, j < 0).set(0)
    j = xp.clip(j, 0, n - 1)
    jp1 = xp.clip(j + 1, 0, n - 1)

    # Broadcast indices to match y shape except for the last axis
    if y.ndim > 1:
        # Create broadcast shape for indices
        broadcast_shape = [*y.shape[:-1], 1]
        j = xp.broadcast_to(j, broadcast_shape)
        jp1 = xp.broadcast_to(jp1, broadcast_shape)
        g = xp.broadcast_to(g, broadcast_shape)

    return (1 - g) * xp.take_along_axis(y, j, axis=-1) + g * xp.take_along_axis(
        y, jp1, axis=-1
    )
