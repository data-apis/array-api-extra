"""
Testing utilities.

Note that this is private API; don't expect it to be stable.
"""

from types import ModuleType

from ._utils._compat import (
    array_namespace,
    is_cupy_namespace,
    is_pydata_sparse_namespace,
    is_torch_namespace,
)
from ._utils._typing import Array

__all__ = ["xp_assert_close", "xp_assert_equal"]


def _check_ns_shape_dtype(
    actual: Array, desired: Array
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

    msg = f"shapes do not match: {actual.shape} != f{desired.shape}"
    assert actual.shape == desired.shape, msg

    msg = f"dtypes do not match: {actual.dtype} != {desired.dtype}"
    assert actual.dtype == desired.dtype, msg

    return desired_xp


def xp_assert_equal(actual: Array, desired: Array, err_msg: str = "") -> None:
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
    """
    xp = _check_ns_shape_dtype(actual, desired)

    if is_cupy_namespace(xp):
        xp.testing.assert_array_equal(actual, desired, err_msg=err_msg)
    elif is_torch_namespace(xp):
        # PyTorch recommends using `rtol=0, atol=0` like this
        # to test for exact equality
        xp.testing.assert_close(
            actual,
            desired,
            rtol=0,
            atol=0,
            equal_nan=True,
            check_dtype=False,
            msg=err_msg or None,
        )
    else:
        import numpy as np  # pylint: disable=import-outside-toplevel

        if is_pydata_sparse_namespace(xp):
            actual = actual.todense()
            desired = desired.todense()

        # JAX uses `np.testing`
        np.testing.assert_array_equal(actual, desired, err_msg=err_msg)


def xp_assert_close(
    actual: Array,
    desired: Array,
    *,
    rtol: float | None = None,
    atol: float = 0,
    err_msg: str = "",
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
    """
    xp = _check_ns_shape_dtype(actual, desired)

    floating = xp.isdtype(actual.dtype, ("real floating", "complex floating"))
    if rtol is None and floating:
        # multiplier of 4 is used as for `np.float64` this puts the default `rtol`
        # roughly half way between sqrt(eps) and the default for
        # `numpy.testing.assert_allclose`, 1e-7
        rtol = xp.finfo(actual.dtype).eps ** 0.5 * 4
    elif rtol is None:
        rtol = 1e-7

    if is_cupy_namespace(xp):
        xp.testing.assert_allclose(
            actual, desired, rtol=rtol, atol=atol, err_msg=err_msg
        )
    elif is_torch_namespace(xp):
        xp.testing.assert_close(
            actual, desired, rtol=rtol, atol=atol, equal_nan=True, msg=err_msg or None
        )
    else:
        import numpy as np  # pylint: disable=import-outside-toplevel

        if is_pydata_sparse_namespace(xp):
            actual = actual.to_dense()
            desired = desired.to_dense()

        # JAX uses `np.testing`
        assert isinstance(rtol, float)
        np.testing.assert_allclose(
            actual, desired, rtol=rtol, atol=atol, err_msg=err_msg
        )
