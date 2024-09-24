from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Array, ModuleType

__all__ = ["atleast_nd"]


def atleast_nd(x: Array, /, *, ndim: int, xp: ModuleType) -> Array:
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


def expand_dims(
    a: Array, /, *, axis: int | tuple[int, ...] = (0,), xp: ModuleType
) -> Array:
    """
    Expand the shape of an array.

    Insert (a) new axis/axes that will appear at the position(s) specified by
    `axis` in the expanded array shape.

    This is ``xp.expand_dims`` for `axis` an int *or a tuple of ints*.
    Equivalent to ``numpy.expand_dims`` for NumPy arrays.

    Parameters
    ----------
    a : array
    axis : int or tuple of ints
        Position(s) in the expanded axes where the new axis (or axes) is/are placed.
        Default: ``(0,)``.
    xp : array_namespace
        The standard-compatible namespace for `a`.

    Returns
    -------
    res : array
        `a` with an expanded shape.

    Examples
    --------
    # >>> import numpy as np
    # >>> x = np.array([1, 2])
    # >>> x.shape
    # (2,)

    # The following is equivalent to ``x[np.newaxis, :]`` or ``x[np.newaxis]``:

    # >>> y = np.expand_dims(x, axis=0)
    # >>> y
    # array([[1, 2]])
    # >>> y.shape
    # (1, 2)

    # The following is equivalent to ``x[:, np.newaxis]``:

    # >>> y = np.expand_dims(x, axis=1)
    # >>> y
    # array([[1],
    #        [2]])
    # >>> y.shape
    # (2, 1)

    # ``axis`` may also be a tuple:

    # >>> y = np.expand_dims(x, axis=(0, 1))
    # >>> y
    # array([[[1, 2]]])

    # >>> y = np.expand_dims(x, axis=(2, 0))
    # >>> y
    # array([[[1],
    #         [2]]])

    # Note that some examples may use ``None`` instead of ``np.newaxis``.  These
    # are the same objects:

    # >>> np.newaxis is None
    # True

    """
    if not isinstance(axis, tuple):
        axis = (axis,)
    for i in axis:
        a = xp.expand_dims(a, axis=i, xp=xp)
    return a


def kron(a: Array, b: Array, /, *, xp: ModuleType) -> Array:
    """
    Kronecker product of two arrays.

    Computes the Kronecker product, a composite array made of blocks of the
    second array scaled by the first.

    Equivalent to ``numpy.kron`` for NumPy arrays.

    Parameters
    ----------
    a, b : array
    xp : array_namespace
        The standard-compatible namespace for `a` and `b`.

    Returns
    -------
    res : array

    Notes
    -----
    The function assumes that the number of dimensions of `a` and `b`
    are the same, if necessary prepending the smallest with ones.
    If ``a.shape = (r0,r1,..,rN)`` and ``b.shape = (s0,s1,...,sN)``,
    the Kronecker product has shape ``(r0*s0, r1*s1, ..., rN*SN)``.
    The elements are products of elements from `a` and `b`, organized
    explicitly by::

        kron(a,b)[k0,k1,...,kN] = a[i0,i1,...,iN] * b[j0,j1,...,jN]

    where::

        kt = it * st + jt,  t = 0,...,N

    In the common 2-D case (N=1), the block structure can be visualized::

        [[ a[0,0]*b,   a[0,1]*b,  ... , a[0,-1]*b  ],
         [  ...                              ...   ],
         [ a[-1,0]*b,  a[-1,1]*b, ... , a[-1,-1]*b ]]


    Examples
    --------
    # >>> import numpy as np
    # >>> np.kron([1,10,100], [5,6,7])
    # array([  5,   6,   7, ..., 500, 600, 700])
    # >>> np.kron([5,6,7], [1,10,100])
    # array([  5,  50, 500, ...,   7,  70, 700])

    # >>> np.kron(np.eye(2), np.ones((2,2)))
    # array([[1.,  1.,  0.,  0.],
    #        [1.,  1.,  0.,  0.],
    #        [0.,  0.,  1.,  1.],
    #        [0.,  0.,  1.,  1.]])

    # >>> a = np.arange(100).reshape((2,5,2,5))
    # >>> b = np.arange(24).reshape((2,3,4))
    # >>> c = np.kron(a,b)
    # >>> c.shape
    # (2, 10, 6, 20)
    # >>> I = (1,3,0,2)
    # >>> J = (0,2,1)
    # >>> J1 = (0,) + J             # extend to ndim=4
    # >>> S1 = (1,) + b.shape
    # >>> K = tuple(np.array(I) * np.array(S1) + np.array(J1))
    # >>> c[K] == a[I]*b[J]
    # True

    """

    b = xp.asarray(b)
    singletons = (1,) * (b.ndim - a.ndim)
    a = xp.broadcast_to(xp.asarray(a), singletons + a.shape)

    nd_b, nd_a = b.ndim, a.ndim
    nd_max = max(nd_b, nd_a)
    if nd_a == 0 or nd_b == 0:
        return xp.multiply(a, b)

    a_shape = a.shape
    b_shape = b.shape

    # Equalise the shapes by prepending smaller one with 1s
    a_shape = (1,) * max(0, nd_b - nd_a) + a_shape
    b_shape = (1,) * max(0, nd_a - nd_b) + b_shape

    # Insert empty dimensions
    a_arr = expand_dims(a, axis=tuple(range(nd_b - nd_a)), xp=xp)
    b_arr = expand_dims(b, axis=tuple(range(nd_a - nd_b)), xp=xp)

    # Compute the product
    a_arr = expand_dims(a_arr, axis=tuple(range(1, nd_max * 2, 2)), xp=xp)
    b_arr = expand_dims(b_arr, axis=tuple(range(0, nd_max * 2, 2)), xp=xp)
    result = xp.multiply(a_arr, b_arr)

    # Reshape back and return
    a_shape = xp.asarray(a_shape)
    b_shape = xp.asarray(b_shape)
    return xp.reshape(result, tuple(xp.multiply(a_shape, b_shape)))
