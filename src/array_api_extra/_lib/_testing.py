"""
Testing utilities.

Note that this is private API; don't expect it to be stable.
See also ..testing for public testing utilities.
"""

import math
from types import ModuleType
from typing import cast

import numpy as np
import pytest

from ._utils._compat import (
    array_namespace,
    is_array_api_strict_namespace,
    is_cupy_namespace,
    is_dask_namespace,
    is_numpy_namespace,
    is_pydata_sparse_namespace,
    is_torch_namespace,
)
from ._utils._typing import Array

__all__ = ["xp_assert_close", "xp_assert_equal"]


def _check_ns_shape_dtype(
    actual: Array,
    desired: Array,
    check_dtype: bool,
    check_shape: bool,
    check_scalar: bool,
) -> ModuleType:  # numpydoc ignore=RT03
    """
    Assert that namespace, shape and dtype of the two arrays match.

    Parameters
    ----------
    actual : Array
        The array produced by the tested function.
    desired : Array
        The expected array (typically hardcoded).

    Returns
    -------
    Arrays namespace.
    """
    actual_xp = array_namespace(actual)  # Raises on scalars and lists
    desired_xp = array_namespace(desired)

    msg = f"namespaces do not match: {actual_xp} != f{desired_xp}"
    assert actual_xp == desired_xp, msg

    if check_shape:
        actual_shape = actual.shape
        desired_shape = desired.shape
        if is_dask_namespace(desired_xp):
            # Dask uses nan instead of None for unknown shapes
            if any(math.isnan(i) for i in cast(tuple[float, ...], actual_shape)):
                actual_shape = actual.compute().shape  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
            if any(math.isnan(i) for i in cast(tuple[float, ...], desired_shape)):
                desired_shape = desired.compute().shape  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]

        msg = f"shapes do not match: {actual_shape} != f{desired_shape}"
        assert actual_shape == desired_shape, msg

    if check_dtype:
        msg = f"dtypes do not match: {actual.dtype} != {desired.dtype}"
        assert actual.dtype == desired.dtype, msg

    if is_numpy_namespace(actual_xp) and check_scalar:
        # only NumPy distinguishes between scalars and arrays; we do if check_scalar.
        _msg = (
            "array-ness does not match:\n Actual: "
            f"{type(actual)}\n Desired: {type(desired)}"
        )
        assert (np.isscalar(actual) and np.isscalar(desired)) or (
            not np.isscalar(actual) and not np.isscalar(desired)
        ), _msg

    return desired_xp


def _prepare_for_test(array: Array, xp: ModuleType) -> Array:
    """
    Ensure that the array can be compared with np.testing.

    This involves transferring it from GPU to CPU memory, densifying it, etc.
    """
    if is_torch_namespace(xp):
        return np.asarray(array.cpu())  # type: ignore[attr-defined, return-value]  # pyright: ignore[reportAttributeAccessIssue, reportUnknownArgumentType, reportReturnType]
    if is_pydata_sparse_namespace(xp):
        return array.todense()  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
    if is_array_api_strict_namespace(xp):
        # Note: we deliberately did not add a `.to_device` method in _typing.pyi
        # even if it is required by the standard as many backends don't support it
        return array.to_device(xp.Device("CPU_DEVICE"))  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
    if is_cupy_namespace(xp):
        return xp.asnumpy(array)
    return array


def xp_assert_equal(
    actual: Array,
    desired: Array,
    *,
    err_msg: str = "",
    check_dtype: bool = True,
    check_shape: bool = True,
    check_scalar: bool = False,
) -> None:
    """
    Array-API compatible version of `np.testing.assert_array_equal`.

    Parameters
    ----------
    actual : Array
        The array produced by the tested function.
    desired : Array
        The expected array (typically hardcoded).
    err_msg : str, optional
        Error message to display on failure.
    check_dtype, check_shape : bool, default: True
        Whether to check agreement between actual and desired dtypes and shapes
    check_scalar : bool, default: False
        NumPy only: whether to check agreement between actual and desired types -
        0d array vs scalar.

    See Also
    --------
    xp_assert_close : Similar function for inexact equality checks.
    numpy.testing.assert_array_equal : Similar function for NumPy arrays.
    """
    xp = _check_ns_shape_dtype(actual, desired, check_dtype, check_shape, check_scalar)
    actual = _prepare_for_test(actual, xp)
    desired = _prepare_for_test(desired, xp)
    np.testing.assert_array_equal(actual, desired, err_msg=err_msg)


def xp_assert_close(
    actual: Array,
    desired: Array,
    *,
    rtol: float | None = None,
    atol: float = 0,
    err_msg: str = "",
    check_dtype: bool = True,
    check_shape: bool = True,
    check_scalar: bool = False,
) -> None:
    """
    Array-API compatible version of `np.testing.assert_allclose`.

    Parameters
    ----------
    actual : Array
        The array produced by the tested function.
    desired : Array
        The expected array (typically hardcoded).
    rtol : float, optional
        Relative tolerance. Default: dtype-dependent.
    atol : float, optional
        Absolute tolerance. Default: 0.
    err_msg : str, optional
        Error message to display on failure.
    check_dtype, check_shape : bool, default: True
        Whether to check agreement between actual and desired dtypes and shapes
    check_scalar : bool, default: False
        NumPy only: whether to check agreement between actual and desired types -
        0d array vs scalar.

    See Also
    --------
    xp_assert_equal : Similar function for exact equality checks.
    isclose : Public function for checking closeness.
    numpy.testing.assert_allclose : Similar function for NumPy arrays.

    Notes
    -----
    The default `atol` and `rtol` differ from `xp.all(xpx.isclose(a, b))`.
    """
    xp = _check_ns_shape_dtype(actual, desired, check_dtype, check_shape, check_scalar)

    floating = xp.isdtype(actual.dtype, ("real floating", "complex floating"))
    if rtol is None and floating:
        # multiplier of 4 is used as for `np.float64` this puts the default `rtol`
        # roughly half way between sqrt(eps) and the default for
        # `numpy.testing.assert_allclose`, 1e-7
        rtol = xp.finfo(actual.dtype).eps ** 0.5 * 4
    elif rtol is None:
        rtol = 1e-7

    actual = _prepare_for_test(actual, xp)
    desired = _prepare_for_test(desired, xp)

    # JAX/Dask arrays work directly with `np.testing`
    assert isinstance(rtol, float)
    np.testing.assert_allclose(  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue]
        actual,  # pyright: ignore[reportArgumentType]
        desired,  # pyright: ignore[reportArgumentType]
        rtol=rtol,
        atol=atol,
        err_msg=err_msg,
    )


def xfail(request: pytest.FixtureRequest, reason: str) -> None:
    """
    XFAIL the currently running test.

    Unlike ``pytest.xfail``, allow rest of test to execute instead of immediately
    halting it, so that it may result in a XPASS.
    xref https://github.com/pandas-dev/pandas/issues/38902

    Parameters
    ----------
    request : pytest.FixtureRequest
        ``request`` argument of the test function.
    reason : str
        Reason for the expected failure.
    """
    request.node.add_marker(pytest.mark.xfail(reason=reason))
