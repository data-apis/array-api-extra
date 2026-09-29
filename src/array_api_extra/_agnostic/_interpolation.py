"""Array-agnostic implementations for interpolation functions."""

import math
from typing import cast

from .._lib import _compat
from .._lib._typing import Array, ArrayNamespace, DType

__all__ = ["interp"]


def _interp_component(
    x: Array,
    x_lo: Array,
    x_hi: Array,
    y_lo: Array,
    y_hi: Array,
    /,
    *,
    inactive: Array,
    xp: ArrayNamespace,
) -> Array:
    """Interpolate one real component without invalid arithmetic."""
    coordinate_infinite = xp.isinf(x_lo) | xp.isinf(x_hi)
    values_finite = xp.isfinite(y_lo) & xp.isfinite(y_hi)
    regular = ~inactive & ~coordinate_infinite & values_finite

    safe_x = xp.where(regular, x, xp.zeros_like(x))
    safe_x_lo = xp.where(regular, x_lo, xp.zeros_like(x_lo))
    safe_x_hi = xp.where(regular, x_hi, xp.ones_like(x_hi))
    safe_y_lo = xp.where(regular, y_lo, xp.zeros_like(y_lo))
    safe_y_hi = xp.where(regular, y_hi, xp.zeros_like(y_hi))
    slope = (safe_y_hi - safe_y_lo) / (safe_x_hi - safe_x_lo)

    device = _compat.device(y_lo)
    nan = xp.asarray(xp.nan, dtype=y_lo.dtype, device=device)
    equal_values = y_lo == y_hi

    left_delta = safe_x - safe_x_lo
    left_invalid = (slope == 0) & xp.isinf(left_delta)
    left_out = (
        xp.where(left_invalid, xp.zeros_like(slope), slope)
        * xp.where(left_invalid, xp.zeros_like(left_delta), left_delta)
        + safe_y_lo
    )
    retry = left_invalid | xp.isnan(left_out)

    right_delta = safe_x - safe_x_hi
    right_invalid = (slope == 0) & xp.isinf(right_delta)
    right_out = (
        xp.where(right_invalid, xp.zeros_like(slope), slope)
        * xp.where(right_invalid, xp.zeros_like(right_delta), right_delta)
        + safe_y_hi
    )
    right_failed = right_invalid | xp.isnan(right_out)
    out = xp.where(retry, right_out, left_out)
    out = xp.where(retry & right_failed, nan, out)
    out = xp.where(retry & right_failed & equal_values, safe_y_lo, out)

    finite_coordinates_nonfinite_values = ~coordinate_infinite & ~values_finite
    nonfinite_value_out = xp.where(
        xp.isfinite(y_lo),
        y_hi,
        xp.where(xp.isfinite(y_hi), y_lo, xp.where(equal_values, y_lo, nan)),
    )
    out = xp.where(finite_coordinates_nonfinite_values, nonfinite_value_out, out)

    only_lo_coordinate_infinite = xp.isinf(x_lo) & ~xp.isinf(x_hi)
    only_hi_coordinate_infinite = ~xp.isinf(x_lo) & xp.isinf(x_hi)
    infinite_coordinate_out = xp.where(
        values_finite & only_lo_coordinate_infinite,
        y_hi,
        xp.where(
            values_finite & only_hi_coordinate_infinite,
            y_lo,
            xp.where(equal_values, y_lo, nan),
        ),
    )
    return xp.where(coordinate_infinite, infinite_coordinate_out, out)


def _combine_complex(
    real: Array, imag: Array, /, *, dtype: DType, xp: ArrayNamespace
) -> Array:
    """Combine real components without multiplying zero by an infinity."""
    device = _compat.device(real)
    real_complex = xp.astype(real, dtype)
    finite_imag = xp.where(xp.isfinite(imag), imag, xp.zeros_like(imag))
    out = real_complex + xp.astype(finite_imag, dtype) * 1j

    positive_inf = xp.asarray(complex(0.0, math.inf), dtype=dtype, device=device)
    negative_inf = xp.asarray(complex(0.0, -math.inf), dtype=dtype, device=device)
    imaginary_nan = xp.asarray(complex(0.0, math.nan), dtype=dtype, device=device)
    out = xp.where(xp.isinf(imag) & (imag > 0), real_complex + positive_inf, out)
    out = xp.where(xp.isinf(imag) & (imag < 0), real_complex + negative_inf, out)
    return xp.where(xp.isnan(imag), real_complex + imaginary_nan, out)


def interp(
    x: Array,
    x_points: Array,
    values: Array,
    /,
    *,
    left: Array | None,
    right: Array | None,
    period: int | float | None,
    xp: ArrayNamespace,
) -> Array:
    # numpydoc ignore=PR01,RT01
    """See docstring in `array_api_extra._interpolation`."""
    if period is not None:
        x = x % period
        x_points = x_points % period
        order = xp.argsort(x_points, stable=True)
        x_points = xp.take(x_points, order, axis=0)
        values = xp.take(values, order, axis=0)
        x_points = xp.concat(
            (x_points[-1:] - period, x_points, x_points[:1] + period), axis=0
        )
        values = xp.concat((values[-1:], values, values[:1]), axis=0)

    x_shape = x.shape
    x_flat = xp.reshape(x, (-1,))
    n_points = cast(int, x_points.shape[0])

    if period is None and n_points == 1:
        out = xp.broadcast_to(values[0], x_flat.shape)
        left_array = values[0] if left is None else left
        right_array = values[-1] if right is None else right
        out = xp.where(x_flat < x_points[0], left_array, out)
        out = xp.where(x_flat > x_points[-1], right_array, out)
        return xp.reshape(out, x_shape)

    right_indices = xp.searchsorted(x_points, x_flat, side="right")
    exact_indices = xp.clip(right_indices - 1, 0, n_points - 1)
    exact_x = xp.take(x_points, exact_indices, axis=0)
    exact_y = xp.take(values, exact_indices, axis=0)
    exact = x_flat == exact_x

    interval_indices = xp.clip(right_indices - 1, 0, n_points - 2)
    x_lo = xp.take(x_points, interval_indices, axis=0)
    x_hi = xp.take(x_points, interval_indices + 1, axis=0)
    y_lo = xp.take(values, interval_indices, axis=0)
    y_hi = xp.take(values, interval_indices + 1, axis=0)

    below = x_flat < x_points[0]
    above = x_flat > x_points[-1]
    query_nan = xp.isnan(x_flat)
    inactive = exact | below | above | query_nan | (x_lo == x_hi)

    if xp.isdtype(values.dtype, "complex floating"):
        out_real = _interp_component(
            x_flat,
            x_lo,
            x_hi,
            xp.real(y_lo),
            xp.real(y_hi),
            inactive=inactive,
            xp=xp,
        )
        out_imag = _interp_component(
            x_flat,
            x_lo,
            x_hi,
            xp.imag(y_lo),
            xp.imag(y_hi),
            inactive=inactive,
            xp=xp,
        )
        exact_real = xp.real(exact_y)
        exact_imag = xp.imag(exact_y)
        left_array = values[0] if left is None else left
        right_array = values[-1] if right is None else right
        out_real = xp.where(exact, exact_real, out_real)
        out_imag = xp.where(exact, exact_imag, out_imag)
        out_real = xp.where(below, xp.real(left_array), out_real)
        out_imag = xp.where(below, xp.imag(left_array), out_imag)
        out_real = xp.where(above, xp.real(right_array), out_real)
        out_imag = xp.where(above, xp.imag(right_array), out_imag)
        nan = xp.asarray(
            xp.nan, dtype=out_real.dtype, device=_compat.device(values)
        )
        out_real = xp.where(query_nan, nan, out_real)
        out_imag = xp.where(query_nan, nan, out_imag)
        out = _combine_complex(out_real, out_imag, dtype=values.dtype, xp=xp)
    else:
        out = _interp_component(
            x_flat, x_lo, x_hi, y_lo, y_hi, inactive=inactive, xp=xp
        )
        out = xp.where(exact, exact_y, out)
        left_array = values[0] if left is None else left
        right_array = values[-1] if right is None else right
        out = xp.where(below, left_array, out)
        out = xp.where(above, right_array, out)
        nan = xp.asarray(xp.nan, dtype=values.dtype, device=_compat.device(values))
        out = xp.where(query_nan, nan, out)

    return xp.reshape(out, x_shape)
