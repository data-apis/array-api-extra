from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Array, ModuleType

__all__ = ["atleast_nd", "cov"]


def atleast_nd(x: Array, *, ndim: int, xp: ModuleType) -> Array:
    """
    Recursively expand the dimension of an array to at least `ndim`.

    Parameters
    ----------
    x : array
    ndim : int
        The minimum number of dimensions for the result.
    xp : array_namespace
        The standard-compatible namespace for `x`.

    Returns
    -------
    res : array
        An array with ``res.ndim`` >= `ndim`.
        If ``x.ndim`` >= `ndim`, `x` is returned.
        If ``x.ndim`` < `ndim`, `x` is expanded by prepending new axes
        until ``res.ndim`` equals `ndim`.

    Examples
    --------
    >>> import array_api_strict as xp
    >>> import array_api_extra as xpx
    >>> x = xp.asarray([1])
    >>> xpx.atleast_nd(x, ndim=3, xp=xp)
    Array([[[1]]], dtype=array_api_strict.int64)

    >>> x = xp.asarray([[[1, 2],
    ...                  [3, 4]]])
    >>> xpx.atleast_nd(x, ndim=1, xp=xp) is x
    True

    """
    if x.ndim < ndim:
        x = xp.expand_dims(x, axis=0)
        x = atleast_nd(x, ndim=ndim, xp=xp)
    return x


def cov(m: Array, /, *, xp: ModuleType) -> Array:
    """
    Estimate a covariance matrix, given data and weights.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element :math:`C_{ij}` is the covariance of
    :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
    of :math:`x_i`.

    This provides a subset of the functionality of ``numpy.cov``.

    Parameters
    ----------
    m : array
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column a single
        observation of all those variables.

    Returns
    -------
    res : array
        The covariance matrix of the variables.

    Examples
    --------
    >>> import numpy as np

    Consider two variables, :math:`x_0` and :math:`x_1`, which
    correlate perfectly, but in opposite directions:

    >>> x = np.array([[0, 2], [1, 1], [2, 0]]).T
    >>> x
    array([[0, 1, 2],
           [2, 1, 0]])

    Note how :math:`x_0` increases while :math:`x_1` decreases. The covariance
    matrix shows this clearly:

    >>> np.cov(x)
    array([[ 1., -1.],
           [-1.,  1.]])

    Note that element :math:`C_{0,1}`, which shows the correlation between
    :math:`x_0` and :math:`x_1`, is negative.

    Further, note how `x` and `y` are combined:

    >>> x = [-2.1, -1,  4.3]
    >>> y = [3,  1.1,  0.12]
    >>> X = np.stack((x, y), axis=0)
    >>> np.cov(X)
    array([[11.71      , -4.286     ], # may vary
           [-4.286     ,  2.144133]])
    >>> np.cov(x)
    array(11.71)
    >>> xp.cov(y)

    """
    m = xp.asarray(m, copy=True)
    dtype = (
        xp.float64 if xp.isdtype(m.dtype, "integral") else xp.result_type(m, xp.float64)
    )

    m = atleast_nd(m, ndim=2, xp=xp)
    m = xp.astype(m, dtype)

    avg = mean(m, axis=1, xp=xp)
    fact = m.shape[1] - 1

    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=2)
        fact = 0.0

    m -= avg[:, None]
    m_transpose = m.T
    if xp.isdtype(m_transpose.dtype, "complex floating"):
        m_transpose = xp.conj(m_transpose)
    c = m @ m_transpose
    c /= fact
    axes = tuple(axis for axis, length in enumerate(c.shape) if length == 1)
    return xp.squeeze(c, axis=axes)


def mean(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    xp: ModuleType,
) -> Array:
    """
    Calculates the arithmetic mean of the input array ``x``.

    In addition to the standard ``mean``, this function supports complex-valued input.

    Parameters
    ----------
    x: array
        input array. Should have a floating-point data type.
    axis: int or tuple of ints, optional
        axis or axes along which arithmetic means must be computed.
        By default, the mean must be computed over the entire array.
        If a tuple of integers, arithmetic means must be computed over multiple axes.
        Default: ``None``.
    keepdims: bool, optional
        if ``True``, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with
        the input array (see :ref:`broadcasting`).
        Otherwise, if ``False``, the reduced axes (dimensions) must not be included
        in the result. Default: ``False``.

    Returns
    -------
    out: array
        if the arithmetic mean was computed over the entire array,
        a zero-dimensional array containing the arithmetic mean;
        otherwise, a non-zero-dimensional array containing the arithmetic means.
        The returned array must have the same data type as ``x``.

    Notes
    -----

    **Special Cases**

    Let ``N`` equal the number of elements over which to compute the arithmetic mean.

    -   If ``N`` is ``0``, the arithmetic mean is ``NaN``.
    -   If ``x_i`` is ``NaN``, the arithmetic mean is ``NaN``
        (i.e., ``NaN`` values propagate).
    """
    if xp.isdtype(x.dtype, "complex floating"):
        x_real = xp.real(x)
        x_imag = xp.imag(x)
        mean_real = xp.mean(x_real, axis=axis, keepdims=keepdims)
        mean_imag = xp.mean(x_imag, axis=axis, keepdims=keepdims)
        return mean_real + (mean_imag * xp.asarray(1j))
    return xp.mean(x, axis=axis, keepdims=keepdims)
