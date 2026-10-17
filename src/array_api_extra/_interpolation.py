"""Delegation layer for interpolation functions."""

import math
from numbers import Number, Real

from . import _agnostic
from ._lib import _compat, _helpers
from ._lib._typing import Array, ArrayNamespace, Device, DType

__all__ = ["interp"]


def _is_python_real_scalar(x: object, /) -> bool:
    if x is True or x is False:
        return False
    return _helpers.is_python_scalar(x) and isinstance(x, Real)


def _is_real_scalar(x: object, /) -> bool:
    if x is True or x is False:
        return False
    return isinstance(x, Real)


def _same_namespace(xp1: ArrayNamespace, xp2: ArrayNamespace, /) -> bool:
    predicates = (
        _compat.is_array_api_strict_namespace,
        _compat.is_cupy_namespace,
        _compat.is_dask_namespace,
        _compat.is_jax_namespace,
        _compat.is_numpy_namespace,
        _compat.is_pydata_sparse_namespace,
        _compat.is_torch_namespace,
    )
    return xp1 is xp2 or any(
        predicate(xp1) and predicate(xp2) for predicate in predicates
    )


def _require_dtype(xp: ArrayNamespace, name: str, /, *, device: Device) -> DType:
    try:
        dtype = xp.__array_namespace_info__().dtypes(device=device)[name]
    except (AttributeError, KeyError, TypeError) as error:
        msg = f"`interp` requires {name} support on the selected device."
        raise TypeError(msg) from error
    return dtype


def _astype_required(
    x: Array, dtype: DType, /, *, name: str, xp: ArrayNamespace
) -> Array:
    out = xp.astype(x, dtype, copy=False)
    if out.dtype != dtype:
        msg = f"`interp` requires {dtype!s} support for `{name}`."
        raise TypeError(msg)
    return out


def _validate_bound(
    bound: Array | complex | None,
    /,
    *,
    name: str,
    values_are_complex: bool,
    xp: ArrayNamespace,
) -> None:
    if bound is None:
        return

    if _compat.is_array_api_obj(bound):
        if bound.ndim != 0:
            msg = f"`{name}` must be a numerical scalar or a 0-dimensional array."
            raise ValueError(msg)
        if not xp.isdtype(
            bound.dtype, ("integral", "real floating", "complex floating")
        ):
            msg = f"`{name}` must have a real or complex numeric dtype."
            raise TypeError(msg)
        bound_is_complex = xp.isdtype(bound.dtype, "complex floating")
    else:
        if bound is True or bound is False or not isinstance(bound, Number):
            msg = f"`{name}` must be a numerical scalar or a 0-dimensional array."
            raise TypeError(msg)
        bound_is_complex = isinstance(bound, complex) and not isinstance(bound, Real)

    if bound_is_complex and not values_are_complex:
        msg = f"`{name}` must be real when `values` is real-valued."
        raise TypeError(msg)


def _as_bound_array(
    bound: Array | complex | None,
    /,
    *,
    dtype: DType,
    device: Device,
    name: str,
    xp: ArrayNamespace,
) -> Array | None:
    if bound is None:
        return None
    if _compat.is_array_api_obj(bound):
        return _astype_required(bound, dtype, name=name, xp=xp)
    out = xp.asarray(bound, dtype=dtype, device=device)
    if out.dtype != dtype:
        msg = f"`interp` requires {dtype!s} support for `{name}`."
        raise TypeError(msg)
    return out


def interp(
    x: Array | float,
    x_points: Array,
    values: Array,
    /,
    *,
    left: Array | complex | None = None,
    right: Array | complex | None = None,
    period: float | Real | None = None,
    xp: ArrayNamespace | None = None,
) -> Array:
    """
    One-dimensional piecewise linear interpolation.

    Evaluate the piecewise linear function defined by sample coordinates
    `x_points` and sample `values` at the query coordinates `x`.

    Parameters
    ----------
    x : Array or real scalar
        Query coordinates. Arrays may have any shape and must have an integral or
        real floating-point dtype. A scalar must be a Python real scalar. Boolean
        values are not supported.
    x_points : Array
        One-dimensional, nonempty sample coordinates with an integral or real
        floating-point dtype. When `period` is ``None``, coordinates must be in
        non-decreasing order; this precondition is not checked. NaN sample
        coordinates are not supported.
    values : Array
        One-dimensional sample values with a real or complex numeric dtype. Its
        length must match `x_points`.
    left : numerical scalar or 0-dimensional Array, optional
        Value returned for queries below the first sample coordinate. By default,
        the first element of `values` is used. Ignored, without validation, when
        `period` is provided.
    right : numerical scalar or 0-dimensional Array, optional
        Value returned for queries above the last sample coordinate. By default,
        the last element of `values` is used. Ignored, without validation, when
        `period` is provided.
    period : real scalar, optional
        Period for the sample and query coordinates. It must remain finite and
        nonzero when converted to ``float64``. A negative value is treated as its
        absolute value. When provided, coordinates are normalized to the period
        and samples are sorted by normalized coordinate.
    xp : array_namespace, optional
        The standard-compatible namespace for the array arguments. Default: infer.

    Returns
    -------
    Array
        Interpolated values with the same shape as `x`. A scalar query produces a
        0-dimensional array. Coordinates are evaluated in ``float64``. Real sample
        values produce ``float64`` output and complex sample values produce
        ``complex128`` output.

    Notes
    -----
    Without `period`, an exact repeated sample coordinate uses the last corresponding
    value. Ties between coordinates that become equal after periodic normalization
    are backend-dependent. The selected backend and device must support ``float64``
    and, for complex `values`, ``complex128``. Native NumPy and CuPy implementations
    are used when available; other namespaces use the array-agnostic implementation.

    Examples
    --------
    >>> import array_api_extra as xpx
    >>> import array_api_strict as xp
    >>> x_points = xp.asarray([0.0, 1.0, 2.0])
    >>> values = xp.asarray([0.0, 10.0, 20.0])
    >>> xpx.interp(xp.asarray([0.5, 1.5]), x_points, values, xp=xp)
    Array([ 5., 15.], dtype=array_api_strict.float64)
    """
    if not _compat.is_array_api_obj(x_points):
        msg = "`x_points` must be an array."
        raise TypeError(msg)
    if not _compat.is_array_api_obj(values):
        msg = "`values` must be an array."
        raise TypeError(msg)

    x_is_scalar = _is_python_real_scalar(x)
    x_is_array = _compat.is_array_api_obj(x)
    if not x_is_scalar and not x_is_array:
        msg = "`x` must be an array or a Python real scalar."
        raise TypeError(msg)

    if period is not None:
        if not _is_real_scalar(period):
            msg = "`period` must be a finite real scalar or None."
            raise TypeError(msg)
        try:
            period = float(abs(period))
        except (OverflowError, ValueError) as error:
            msg = "`period` must be representable as a finite float."
            raise ValueError(msg) from error
        if not math.isfinite(period):
            msg = "`period` must be finite after conversion to float."
            raise ValueError(msg)
        if period == 0:
            msg = "`period` must be nonzero after conversion to float."
            raise ValueError(msg)

    namespace_args: list[Array] = [x_points, values]
    if _compat.is_array_api_obj(x):
        namespace_args.append(x)
    if period is None:
        if _compat.is_array_api_obj(left):
            namespace_args.append(left)
        if _compat.is_array_api_obj(right):
            namespace_args.append(right)
    inferred_xp = _compat.array_namespace(*namespace_args)
    if xp is None:
        xp = inferred_xp
    elif not _same_namespace(xp, inferred_xp):
        msg = "`xp` must match the namespace of the array arguments."
        raise TypeError(msg)

    if x_points.ndim != 1:
        msg = "`x_points` must be one-dimensional."
        raise ValueError(msg)
    if values.ndim != 1:
        msg = "`values` must be one-dimensional."
        raise ValueError(msg)
    (n_points,) = _helpers.eager_shape(x_points)
    (n_values,) = _helpers.eager_shape(values)
    if n_points == 0:
        msg = "`x_points` and `values` must be nonempty."
        raise ValueError(msg)
    if n_points != n_values:
        msg = "`x_points` and `values` must have the same length."
        raise ValueError(msg)

    if not xp.isdtype(x_points.dtype, ("integral", "real floating")):
        msg = "`x_points` must have an integral or real floating-point dtype."
        raise TypeError(msg)
    if _compat.is_array_api_obj(x) and not xp.isdtype(
        x.dtype, ("integral", "real floating")
    ):
        msg = "`x` must have an integral or real floating-point dtype."
        raise TypeError(msg)
    if not xp.isdtype(values.dtype, ("integral", "real floating", "complex floating")):
        msg = "`values` must have a real or complex numeric dtype."
        raise TypeError(msg)

    values_are_complex = xp.isdtype(values.dtype, "complex floating")
    if period is None:
        _validate_bound(left, name="left", values_are_complex=values_are_complex, xp=xp)
        _validate_bound(
            right, name="right", values_are_complex=values_are_complex, xp=xp
        )

    arrays: list[Array] = [x_points, values]
    if _compat.is_array_api_obj(x):
        arrays.append(x)
    if period is None:
        if _compat.is_array_api_obj(left):
            arrays.append(left)
        if _compat.is_array_api_obj(right):
            arrays.append(right)
    device = _compat.device(x_points)
    if any(_compat.device(array) != device for array in arrays[1:]):
        msg = "All array arguments must be on the same device."
        raise ValueError(msg)

    coordinate_dtype = _require_dtype(xp, "float64", device=device)
    value_dtype = _require_dtype(
        xp, "complex128" if values_are_complex else "float64", device=device
    )
    x_points = _astype_required(x_points, coordinate_dtype, name="x_points", xp=xp)
    if _compat.is_array_api_obj(x):
        x_array = _astype_required(x, coordinate_dtype, name="x", xp=xp)
    else:
        x_array = xp.asarray(x, dtype=coordinate_dtype, device=device)
        if x_array.dtype != coordinate_dtype:
            msg = "`interp` requires float64 support for `x`."
            raise TypeError(msg)
    values = _astype_required(values, value_dtype, name="values", xp=xp)

    if period is None:
        left_array = _as_bound_array(
            left, dtype=value_dtype, device=device, name="left", xp=xp
        )
        right_array = _as_bound_array(
            right, dtype=value_dtype, device=device, name="right", xp=xp
        )
    else:
        left_array = right_array = None

    native_complex_bounds_unsupported = (
        _compat.is_numpy_namespace(xp)
        and values_are_complex
        and (left_array is not None or right_array is not None)
    )
    if (
        _compat.is_numpy_namespace(xp) or _compat.is_cupy_namespace(xp)
    ) and not native_complex_bounds_unsupported:
        out = xp.interp(
            x_array,
            x_points,
            values,
            left=left_array,
            right=right_array,
            period=period,
        )
        if x_array.ndim == 0:
            out = xp.asarray(out, dtype=value_dtype, device=device)
        return out

    return _agnostic._interpolation.interp(
        x_array,
        x_points,
        values,
        left=left_array,
        right=right_array,
        period=period,
        xp=xp,
    )
