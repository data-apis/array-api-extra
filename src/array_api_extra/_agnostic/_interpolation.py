"""Array-agnostic implementations for interpolation functions."""

import math
from typing import cast

from .._lib import _compat
from .._lib._typing import Array, ArrayNamespace, DType

__all__ = ["interp"]


def _safe_difference(x1: Array, x2: Array, /, *, xp: ArrayNamespace) -> Array:
    """Subtract finite arrays with IEEE overflow results but without warnings."""
    zero = xp.zeros_like(x1)
    maximum = xp.asarray(
        xp.finfo(x1.dtype).max, dtype=x1.dtype, device=_compat.device(x1)
    )
    negative_x2 = xp.where(x2 < 0, x2, zero)
    positive_x2 = xp.where(x2 > 0, x2, zero)
    positive_overflow = (x1 > 0) & (x2 < 0) & (x1 > maximum + negative_x2)
    negative_overflow = (x1 < 0) & (x2 > 0) & (x1 < -maximum + positive_x2)
    overflow = positive_overflow | negative_overflow

    out = xp.where(overflow, zero, x1) - xp.where(overflow, zero, x2)
    infinity = xp.asarray(math.inf, dtype=x1.dtype, device=_compat.device(x1))
    out = xp.where(positive_overflow, infinity, out)
    return xp.where(negative_overflow, -infinity, out)


def _safe_divide(
    numerator: Array, denominator: Array, /, *, xp: ArrayNamespace
) -> Array:
    """Divide finite arrays with IEEE overflow results but without warnings."""
    zero = xp.zeros_like(numerator)
    one = xp.ones_like(denominator)
    maximum = xp.asarray(
        xp.finfo(numerator.dtype).max,
        dtype=numerator.dtype,
        device=_compat.device(numerator),
    )
    absolute_denominator = xp.abs(denominator)
    small_denominator = absolute_denominator < 1
    overflow_threshold = maximum * xp.where(
        small_denominator, absolute_denominator, zero
    )
    overflow = (
        (numerator != 0)
        & (denominator != 0)
        & small_denominator
        & (xp.abs(numerator) > overflow_threshold)
    )

    out = xp.where(overflow, zero, numerator) / xp.where(overflow, one, denominator)
    infinity = xp.asarray(
        math.inf, dtype=numerator.dtype, device=_compat.device(numerator)
    )
    positive_overflow = overflow & ((numerator > 0) == (denominator > 0))
    out = xp.where(positive_overflow, infinity, out)
    return xp.where(overflow & ~positive_overflow, -infinity, out)


def _safe_multiply(
    x1: Array, x2: Array, /, *, zero_times_infinity: float, xp: ArrayNamespace
) -> Array:
    """Multiply arrays without warnings from overflow or zero times infinity."""
    zero = xp.zeros_like(x1)
    one = xp.ones_like(x2)
    maximum = xp.asarray(
        xp.finfo(x1.dtype).max, dtype=x1.dtype, device=_compat.device(x1)
    )
    absolute_x1 = xp.abs(x1)
    absolute_x2 = xp.abs(x2)
    large_x2 = absolute_x2 > 1
    overflow_threshold = maximum / xp.where(large_x2, absolute_x2, one)
    overflow = (
        xp.isfinite(x1)
        & xp.isfinite(x2)
        & large_x2
        & (absolute_x1 > overflow_threshold)
    )
    zero_inf = ((x1 == 0) & xp.isinf(x2)) | (xp.isinf(x1) & (x2 == 0))
    suppressed = overflow | zero_inf

    out = xp.where(suppressed, zero, x1) * xp.where(suppressed, zero, x2)
    infinity = xp.asarray(math.inf, dtype=x1.dtype, device=_compat.device(x1))
    positive_overflow = overflow & ((x1 > 0) == (x2 > 0))
    out = xp.where(positive_overflow, infinity, out)
    out = xp.where(overflow & ~positive_overflow, -infinity, out)
    zero_inf_value = xp.asarray(
        zero_times_infinity, dtype=x1.dtype, device=_compat.device(x1)
    )
    return xp.where(zero_inf, zero_inf_value, out)


def _interp_component(
    x: Array,
    x_lo: Array,
    x_hi: Array,
    y_lo: Array,
    y_hi: Array,
    /,
    *,
    inactive: Array,
    reciprocal_first: bool = False,
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
    device = _compat.device(y_lo)
    nan = xp.asarray(math.nan, dtype=y_lo.dtype, device=device)
    equal_values = y_lo == y_hi

    coordinate_difference = _safe_difference(safe_x_hi, safe_x_lo, xp=xp)
    value_difference = _safe_difference(safe_y_hi, safe_y_lo, xp=xp)
    indeterminate_slope = xp.isinf(coordinate_difference) & xp.isinf(value_difference)
    safe_value_difference = xp.where(
        indeterminate_slope, xp.zeros_like(value_difference), value_difference
    )
    safe_coordinate_difference = xp.where(
        indeterminate_slope,
        xp.ones_like(coordinate_difference),
        coordinate_difference,
    )
    if reciprocal_first:
        inverse_coordinate_difference = _safe_divide(
            xp.ones_like(safe_coordinate_difference),
            safe_coordinate_difference,
            xp=xp,
        )
        slope = _safe_multiply(
            safe_value_difference,
            inverse_coordinate_difference,
            zero_times_infinity=math.nan,
            xp=xp,
        )
    else:
        slope = _safe_divide(safe_value_difference, safe_coordinate_difference, xp=xp)
    slope = xp.where(indeterminate_slope, nan, slope)

    left_delta = _safe_difference(safe_x, safe_x_lo, xp=xp)
    left_invalid = (slope == 0) & xp.isinf(left_delta)
    left_out = (
        xp.where(left_invalid, xp.zeros_like(slope), slope)
        * xp.where(left_invalid, xp.zeros_like(left_delta), left_delta)
        + safe_y_lo
    )
    retry = left_invalid | xp.isnan(left_out)

    right_delta = _safe_difference(safe_x, safe_x_hi, xp=xp)
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
    finite_y_lo = xp.where(values_finite, y_lo, xp.zeros_like(y_lo))
    finite_y_hi = xp.where(values_finite, y_hi, xp.zeros_like(y_hi))
    infinite_coordinate_value_overflow = values_finite & xp.isinf(
        _safe_difference(finite_y_hi, finite_y_lo, xp=xp)
    )
    infinite_coordinate_out = xp.where(
        infinite_coordinate_value_overflow,
        nan,
        xp.where(
            values_finite & only_lo_coordinate_infinite,
            y_hi,
            xp.where(
                values_finite & only_hi_coordinate_infinite,
                y_lo,
                xp.where(equal_values, y_lo, nan),
            ),
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
    period: float | None,
    xp: ArrayNamespace,
) -> Array:
    # numpydoc ignore=PR01,RT01
    """See docstring in `array_api_extra._interpolation`."""
    if period is not None:
        x = x % period
        x_points = x_points % period
        order = xp.argsort(x_points, stable=True)
        x_points = xp.take(x_points, order, axis=0)
        if xp.isdtype(values.dtype, "complex floating"):
            values = _combine_complex(
                xp.take(xp.real(values), order, axis=0),
                xp.take(xp.imag(values), order, axis=0),
                dtype=values.dtype,
                xp=xp,
            )
        else:
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
    exact = x_flat == exact_x

    interval_indices = xp.clip(right_indices - 1, 0, n_points - 2)
    x_lo = xp.take(x_points, interval_indices, axis=0)
    x_hi = xp.take(x_points, interval_indices + 1, axis=0)

    below = x_flat < x_points[0]
    above = x_flat > x_points[-1]
    query_nan = xp.isnan(x_flat)
    inactive = exact | below | above | query_nan | (x_lo == x_hi)

    if xp.isdtype(values.dtype, "complex floating"):
        values_real = xp.real(values)
        values_imag = xp.imag(values)
        exact_real = xp.take(values_real, exact_indices, axis=0)
        exact_imag = xp.take(values_imag, exact_indices, axis=0)
        y_lo_real = xp.take(values_real, interval_indices, axis=0)
        y_hi_real = xp.take(values_real, interval_indices + 1, axis=0)
        y_lo_imag = xp.take(values_imag, interval_indices, axis=0)
        y_hi_imag = xp.take(values_imag, interval_indices + 1, axis=0)
        out_real = _interp_component(
            x_flat,
            x_lo,
            x_hi,
            y_lo_real,
            y_hi_real,
            inactive=inactive,
            reciprocal_first=True,
            xp=xp,
        )
        out_imag = _interp_component(
            x_flat,
            x_lo,
            x_hi,
            y_lo_imag,
            y_hi_imag,
            inactive=inactive,
            reciprocal_first=True,
            xp=xp,
        )
        left_array = values[0] if left is None else left
        right_array = values[-1] if right is None else right
        out_real = xp.where(exact, exact_real, out_real)
        out_imag = xp.where(exact, exact_imag, out_imag)
        out_real = xp.where(below, xp.real(left_array), out_real)
        out_imag = xp.where(below, xp.imag(left_array), out_imag)
        out_real = xp.where(above, xp.real(right_array), out_real)
        out_imag = xp.where(above, xp.imag(right_array), out_imag)
        nan = xp.asarray(math.nan, dtype=out_real.dtype, device=_compat.device(values))
        out_real = xp.where(query_nan, nan, out_real)
        out_imag = xp.where(query_nan, xp.zeros_like(out_imag), out_imag)
        out = _combine_complex(out_real, out_imag, dtype=values.dtype, xp=xp)
    else:
        exact_y = xp.take(values, exact_indices, axis=0)
        y_lo = xp.take(values, interval_indices, axis=0)
        y_hi = xp.take(values, interval_indices + 1, axis=0)
        out = _interp_component(
            x_flat, x_lo, x_hi, y_lo, y_hi, inactive=inactive, xp=xp
        )
        out = xp.where(exact, exact_y, out)
        left_array = values[0] if left is None else left
        right_array = values[-1] if right is None else right
        out = xp.where(below, left_array, out)
        out = xp.where(above, right_array, out)
        nan = xp.asarray(math.nan, dtype=values.dtype, device=_compat.device(values))
        out = xp.where(query_nan, nan, out)

    return xp.reshape(out, x_shape)
