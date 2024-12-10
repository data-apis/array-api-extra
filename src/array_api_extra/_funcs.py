from __future__ import annotations  # https://github.com/pylint-dev/pylint/pull/9990

import operator
import typing
import warnings

# https://github.com/pylint-dev/pylint/issues/10112
from collections.abc import Callable  # pylint: disable=import-error
from typing import ClassVar, Literal

from ._lib import _utils
from ._lib._compat import (
    array_namespace,
    is_array_api_obj,
    is_dask_array,
    is_writeable_array,
)

if typing.TYPE_CHECKING:
    from ._lib._typing import Array, Index, ModuleType, Untyped

__all__ = [
    "at",
    "atleast_nd",
    "cov",
    "create_diagonal",
    "expand_dims",
    "kron",
    "setdiff1d",
    "sinc",
]


def atleast_nd(x: Array, /, *, ndim: int, xp: ModuleType | None = None) -> Array:
    """
    Recursively expand the dimension of an array to at least `ndim`.

    Parameters
    ----------
    x : array
    ndim : int
        The minimum number of dimensions for the result.
    xp : array_namespace, optional
        The standard-compatible namespace for `x`. Default: infer

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
    if xp is None:
        xp = array_namespace(x)

    if x.ndim < ndim:
        x = xp.expand_dims(x, axis=0)
        x = atleast_nd(x, ndim=ndim, xp=xp)
    return x


def cov(m: Array, /, *, xp: ModuleType | None = None) -> Array:
    """
    Estimate a covariance matrix.

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
    xp : array_namespace, optional
        The standard-compatible namespace for `m`. Default: infer

    Returns
    -------
    res : array
        The covariance matrix of the variables.

    Examples
    --------
    >>> import array_api_strict as xp
    >>> import array_api_extra as xpx

    Consider two variables, :math:`x_0` and :math:`x_1`, which
    correlate perfectly, but in opposite directions:

    >>> x = xp.asarray([[0, 2], [1, 1], [2, 0]]).T
    >>> x
    Array([[0, 1, 2],
           [2, 1, 0]], dtype=array_api_strict.int64)

    Note how :math:`x_0` increases while :math:`x_1` decreases. The covariance
    matrix shows this clearly:

    >>> xpx.cov(x, xp=xp)
    Array([[ 1., -1.],
           [-1.,  1.]], dtype=array_api_strict.float64)


    Note that element :math:`C_{0,1}`, which shows the correlation between
    :math:`x_0` and :math:`x_1`, is negative.

    Further, note how `x` and `y` are combined:

    >>> x = xp.asarray([-2.1, -1,  4.3])
    >>> y = xp.asarray([3,  1.1,  0.12])
    >>> X = xp.stack((x, y), axis=0)
    >>> xpx.cov(X, xp=xp)
    Array([[11.71      , -4.286     ],
           [-4.286     ,  2.14413333]], dtype=array_api_strict.float64)

    >>> xpx.cov(x, xp=xp)
    Array(11.71, dtype=array_api_strict.float64)

    >>> xpx.cov(y, xp=xp)
    Array(2.14413333, dtype=array_api_strict.float64)

    """
    if xp is None:
        xp = array_namespace(m)

    m = xp.asarray(m, copy=True)
    dtype = (
        xp.float64 if xp.isdtype(m.dtype, "integral") else xp.result_type(m, xp.float64)
    )

    m = atleast_nd(m, ndim=2, xp=xp)
    m = xp.astype(m, dtype)

    avg = _utils.mean(m, axis=1, xp=xp)
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


def create_diagonal(
    x: Array, /, *, offset: int = 0, xp: ModuleType | None = None
) -> Array:
    """
    Construct a diagonal array.

    Parameters
    ----------
    x : array
        A 1-D array
    offset : int, optional
        Offset from the leading diagonal (default is ``0``).
        Use positive ints for diagonals above the leading diagonal,
        and negative ints for diagonals below the leading diagonal.
    xp : array_namespace, optional
        The standard-compatible namespace for `x`. Default: infer

    Returns
    -------
    res : array
        A 2-D array with `x` on the diagonal (offset by `offset`).

    Examples
    --------
    >>> import array_api_strict as xp
    >>> import array_api_extra as xpx
    >>> x = xp.asarray([2, 4, 8])

    >>> xpx.create_diagonal(x, xp=xp)
    Array([[2, 0, 0],
           [0, 4, 0],
           [0, 0, 8]], dtype=array_api_strict.int64)

    >>> xpx.create_diagonal(x, offset=-2, xp=xp)
    Array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [2, 0, 0, 0, 0],
           [0, 4, 0, 0, 0],
           [0, 0, 8, 0, 0]], dtype=array_api_strict.int64)

    """
    if xp is None:
        xp = array_namespace(x)

    if x.ndim != 1:
        err_msg = "`x` must be 1-dimensional."
        raise ValueError(err_msg)
    n = x.shape[0] + abs(offset)
    diag = xp.zeros(n**2, dtype=x.dtype, device=x.device)
    i = offset if offset >= 0 else abs(offset) * n
    diag[i : min(n * (n - offset), diag.shape[0]) : n + 1] = x
    return xp.reshape(diag, (n, n))


def expand_dims(
    a: Array, /, *, axis: int | tuple[int, ...] = (0,), xp: ModuleType | None = None
) -> Array:
    """
    Expand the shape of an array.

    Insert (a) new axis/axes that will appear at the position(s) specified by
    `axis` in the expanded array shape.

    This is ``xp.expand_dims`` for `axis` an int *or a tuple of ints*.
    Roughly equivalent to ``numpy.expand_dims`` for NumPy arrays.

    Parameters
    ----------
    a : array
    axis : int or tuple of ints, optional
        Position(s) in the expanded axes where the new axis (or axes) is/are placed.
        If multiple positions are provided, they should be unique (note that a position
        given by a positive index could also be referred to by a negative index -
        that will also result in an error).
        Default: ``(0,)``.
    xp : array_namespace, optional
        The standard-compatible namespace for `a`. Default: infer

    Returns
    -------
    res : array
        `a` with an expanded shape.

    Examples
    --------
    >>> import array_api_strict as xp
    >>> import array_api_extra as xpx
    >>> x = xp.asarray([1, 2])
    >>> x.shape
    (2,)

    The following is equivalent to ``x[xp.newaxis, :]`` or ``x[xp.newaxis]``:

    >>> y = xpx.expand_dims(x, axis=0, xp=xp)
    >>> y
    Array([[1, 2]], dtype=array_api_strict.int64)
    >>> y.shape
    (1, 2)

    The following is equivalent to ``x[:, xp.newaxis]``:

    >>> y = xpx.expand_dims(x, axis=1, xp=xp)
    >>> y
    Array([[1],
           [2]], dtype=array_api_strict.int64)
    >>> y.shape
    (2, 1)

    ``axis`` may also be a tuple:

    >>> y = xpx.expand_dims(x, axis=(0, 1), xp=xp)
    >>> y
    Array([[[1, 2]]], dtype=array_api_strict.int64)

    >>> y = xpx.expand_dims(x, axis=(2, 0), xp=xp)
    >>> y
    Array([[[1],
            [2]]], dtype=array_api_strict.int64)

    """
    if xp is None:
        xp = array_namespace(a)

    if not isinstance(axis, tuple):
        axis = (axis,)
    ndim = a.ndim + len(axis)
    if axis != () and (min(axis) < -ndim or max(axis) >= ndim):
        err_msg = (
            f"a provided axis position is out of bounds for array of dimension {a.ndim}"
        )
        raise IndexError(err_msg)
    axis = tuple(dim % ndim for dim in axis)
    if len(set(axis)) != len(axis):
        err_msg = "Duplicate dimensions specified in `axis`."
        raise ValueError(err_msg)
    for i in sorted(axis):
        a = xp.expand_dims(a, axis=i)
    return a


def kron(a: Array, b: Array, /, *, xp: ModuleType | None = None) -> Array:
    """
    Kronecker product of two arrays.

    Computes the Kronecker product, a composite array made of blocks of the
    second array scaled by the first.

    Equivalent to ``numpy.kron`` for NumPy arrays.

    Parameters
    ----------
    a, b : array
    xp : array_namespace, optional
        The standard-compatible namespace for `a` and `b`. Default: infer

    Returns
    -------
    res : array
        The Kronecker product of `a` and `b`.

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
    >>> import array_api_strict as xp
    >>> import array_api_extra as xpx
    >>> xpx.kron(xp.asarray([1, 10, 100]), xp.asarray([5, 6, 7]), xp=xp)
    Array([  5,   6,   7,  50,  60,  70, 500,
           600, 700], dtype=array_api_strict.int64)

    >>> xpx.kron(xp.asarray([5, 6, 7]), xp.asarray([1, 10, 100]), xp=xp)
    Array([  5,  50, 500,   6,  60, 600,   7,
            70, 700], dtype=array_api_strict.int64)

    >>> xpx.kron(xp.eye(2), xp.ones((2, 2)), xp=xp)
    Array([[1., 1., 0., 0.],
           [1., 1., 0., 0.],
           [0., 0., 1., 1.],
           [0., 0., 1., 1.]], dtype=array_api_strict.float64)


    >>> a = xp.reshape(xp.arange(100), (2, 5, 2, 5))
    >>> b = xp.reshape(xp.arange(24), (2, 3, 4))
    >>> c = xpx.kron(a, b, xp=xp)
    >>> c.shape
    (2, 10, 6, 20)
    >>> I = (1, 3, 0, 2)
    >>> J = (0, 2, 1)
    >>> J1 = (0,) + J             # extend to ndim=4
    >>> S1 = (1,) + b.shape
    >>> K = tuple(xp.asarray(I) * xp.asarray(S1) + xp.asarray(J1))
    >>> c[K] == a[I]*b[J]
    Array(True, dtype=array_api_strict.bool)

    """
    if xp is None:
        xp = array_namespace(a, b)

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


def setdiff1d(
    x1: Array,
    x2: Array,
    /,
    *,
    assume_unique: bool = False,
    xp: ModuleType | None = None,
) -> Array:
    """
    Find the set difference of two arrays.

    Return the unique values in `x1` that are not in `x2`.

    Parameters
    ----------
    x1 : array
        Input array.
    x2 : array
        Input comparison array.
    assume_unique : bool
        If ``True``, the input arrays are both assumed to be unique, which
        can speed up the calculation. Default is ``False``.
    xp : array_namespace, optional
        The standard-compatible namespace for `x1` and `x2`. Default: infer

    Returns
    -------
    res : array
        1D array of values in `x1` that are not in `x2`. The result
        is sorted when `assume_unique` is ``False``, but otherwise only sorted
        if the input is sorted.

    Examples
    --------
    >>> import array_api_strict as xp
    >>> import array_api_extra as xpx

    >>> x1 = xp.asarray([1, 2, 3, 2, 4, 1])
    >>> x2 = xp.asarray([3, 4, 5, 6])
    >>> xpx.setdiff1d(x1, x2, xp=xp)
    Array([1, 2], dtype=array_api_strict.int64)

    """
    if xp is None:
        xp = array_namespace(x1, x2)

    if assume_unique:
        x1 = xp.reshape(x1, (-1,))
    else:
        x1 = xp.unique_values(x1)
        x2 = xp.unique_values(x2)
    return x1[_utils.in1d(x1, x2, assume_unique=True, invert=True, xp=xp)]


def sinc(x: Array, /, *, xp: ModuleType | None = None) -> Array:
    r"""
    Return the normalized sinc function.

    The sinc function is equal to :math:`\sin(\pi x)/(\pi x)` for any argument
    :math:`x\ne 0`. ``sinc(0)`` takes the limit value 1, making ``sinc`` not
    only everywhere continuous but also infinitely differentiable.

    .. note::

        Note the normalization factor of ``pi`` used in the definition.
        This is the most commonly used definition in signal processing.
        Use ``sinc(x / xp.pi)`` to obtain the unnormalized sinc function
        :math:`\sin(x)/x` that is more common in mathematics.

    Parameters
    ----------
    x : array
        Array (possibly multi-dimensional) of values for which to calculate
        ``sinc(x)``. Must have a real floating point dtype.
    xp : array_namespace, optional
        The standard-compatible namespace for `x`. Default: infer

    Returns
    -------
    res : array
        ``sinc(x)`` calculated elementwise, which has the same shape as the input.

    Notes
    -----
    The name sinc is short for "sine cardinal" or "sinus cardinalis".

    The sinc function is used in various signal processing applications,
    including in anti-aliasing, in the construction of a Lanczos resampling
    filter, and in interpolation.

    For bandlimited interpolation of discrete-time signals, the ideal
    interpolation kernel is proportional to the sinc function.

    References
    ----------
    #. Weisstein, Eric W. "Sinc Function." From MathWorld--A Wolfram Web
       Resource. https://mathworld.wolfram.com/SincFunction.html
    #. Wikipedia, "Sinc function",
       https://en.wikipedia.org/wiki/Sinc_function

    Examples
    --------
    >>> import array_api_strict as xp
    >>> import array_api_extra as xpx
    >>> x = xp.linspace(-4, 4, 41)
    >>> xpx.sinc(x, xp=xp)
    Array([-3.89817183e-17, -4.92362781e-02,
           -8.40918587e-02, -8.90384387e-02,
           -5.84680802e-02,  3.89817183e-17,
            6.68206631e-02,  1.16434881e-01,
            1.26137788e-01,  8.50444803e-02,
           -3.89817183e-17, -1.03943254e-01,
           -1.89206682e-01, -2.16236208e-01,
           -1.55914881e-01,  3.89817183e-17,
            2.33872321e-01,  5.04551152e-01,
            7.56826729e-01,  9.35489284e-01,
            1.00000000e+00,  9.35489284e-01,
            7.56826729e-01,  5.04551152e-01,
            2.33872321e-01,  3.89817183e-17,
           -1.55914881e-01, -2.16236208e-01,
           -1.89206682e-01, -1.03943254e-01,
           -3.89817183e-17,  8.50444803e-02,
            1.26137788e-01,  1.16434881e-01,
            6.68206631e-02,  3.89817183e-17,
           -5.84680802e-02, -8.90384387e-02,
           -8.40918587e-02, -4.92362781e-02,
           -3.89817183e-17], dtype=array_api_strict.float64)

    """
    if xp is None:
        xp = array_namespace(x)

    if not xp.isdtype(x.dtype, "real floating"):
        err_msg = "`x` must have a real floating data type."
        raise ValueError(err_msg)
    # no scalars in `where` - array-api#807
    y = xp.pi * xp.where(
        xp.astype(x, xp.bool),
        x,
        xp.asarray(xp.finfo(x.dtype).eps, dtype=x.dtype, device=x.device),
    )
    return xp.sin(y) / y


_undef = object()


class at:  # pylint: disable=invalid-name
    """
    Update operations for read-only arrays.

    This implements ``jax.numpy.ndarray.at`` for all backends.

    Parameters
    ----------
    x : array
        Input array.
    idx : index, optional
        You may use two alternate syntaxes::

          at(x, idx).set(value)  # or get(), add(), etc.
          at(x)[idx].set(value)

    copy : bool, optional
        True (default)
            Ensure that the inputs are not modified.
        False
            Ensure that the update operation writes back to the input.
            Raise ValueError if a copy cannot be avoided.
        None
            The array parameter *may* be modified in place if it is possible and
            beneficial for performance.
            You should not reuse it after calling this function.
    xp : array_namespace, optional
        The standard-compatible namespace for `x`. Default: infer

    **kwargs:
        If the backend supports an `at` method, any additional keyword
        arguments are passed to it verbatim; e.g. this allows passing
        ``indices_are_sorted=True`` to JAX.

    Returns
    -------
    Updated input array.

    Examples
    --------
    Given either of these equivalent expressions::

      x = at(x)[1].add(2, copy=None)
      x = at(x, 1).add(2, copy=None)

    If x is a JAX array, they are the same as::

      x = x.at[1].add(2)

    If x is a read-only numpy array, they are the same as::

      x = x.copy()
      x[1] += 2

    Otherwise, they are the same as::

      x[1] += 2

    Warning
    -------
    When you use copy=None, you should always immediately overwrite
    the parameter array::

        x = at(x, 0).set(2, copy=None)

    The anti-pattern below must be avoided, as it will result in different behaviour
    on read-only versus writeable arrays::

        x = xp.asarray([0, 0, 0])
        y = at(x, 0).set(2, copy=None)
        z = at(x, 1).set(3, copy=None)

    In the above example, ``x == [0, 0, 0]``, ``y == [2, 0, 0]`` and z == ``[0, 3, 0]``
    when x is read-only, whereas ``x == y == z == [2, 3, 0]`` when x is writeable!

    Warning
    -------
    The array API standard does not support integer array indices.
    The behaviour of update methods when the index is an array of integers
    is undefined; this is particularly true when the index contains multiple
    occurrences of the same index, e.g. ``at(x, [0, 0]).set(2)``.

    Note
    ----
    `sparse <https://sparse.pydata.org/>`_ is not supported by update methods yet.

    See Also
    --------
    `jax.numpy.ndarray.at <https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html>`_
    """

    x: Array
    idx: Index
    __slots__: ClassVar[tuple[str, str]] = ("idx", "x")

    def __init__(self, x: Array, idx: Index = _undef, /) -> None:
        self.x = x
        self.idx = idx

    def __getitem__(self, idx: Index, /) -> at:
        """Allow for the alternate syntax ``at(x)[start:stop:step]``,
        which looks prettier than ``at(x, slice(start, stop, step))``
        and feels more intuitive coming from the JAX documentation.
        """
        if self.idx is not _undef:
            msg = "Index has already been set"
            raise ValueError(msg)
        self.idx = idx
        return self

    def _common(
        self,
        at_op: str,
        y: Array = _undef,
        /,
        copy: bool | None = True,
        xp: ModuleType | None = None,
        _is_update: bool = True,
        **kwargs: Untyped,
    ) -> tuple[Untyped, None] | tuple[None, Array]:
        """Perform common prepocessing.

        Returns
        -------
        If the operation can be resolved by at[], (return value, None)
        Otherwise, (None, preprocessed x)
        """
        if self.idx is _undef:
            msg = (
                "Index has not been set.\n"
                "Usage: either\n"
                "    at(x, idx).set(value)\n"
                "or\n"
                "    at(x)[idx].set(value)\n"
                "(same for all other methods)."
            )
            raise TypeError(msg)

        x = self.x

        if copy is None:
            writeable = is_writeable_array(x)
            copy = _is_update and not writeable
        elif copy:
            writeable = None
        else:
            writeable = is_writeable_array(x)
            if not writeable:
                msg = "Cannot modify parameter in place"
                raise ValueError(msg)

        if copy:
            try:
                at_ = x.at
            except AttributeError:
                # Emulate at[] behaviour for non-JAX arrays
                # with a copy followed by an update
                if xp is None:
                    xp = array_namespace(x)
                # Create writeable copy of read-only numpy array
                x = xp.asarray(x, copy=True)
                if writeable is False:
                    # A copy of a read-only numpy array is writeable
                    writeable = None
            else:
                # Use JAX's at[] or other library that with the same duck-type API
                args = (y,) if y is not _undef else ()
                return getattr(at_[self.idx], at_op)(*args, **kwargs), None

        if _is_update:
            if writeable is None:
                writeable = is_writeable_array(x)
            if not writeable:
                # sparse crashes here
                msg = f"Array {x} has no `at` method and is read-only"
                raise ValueError(msg)

        return None, x

    def get(self, **kwargs: Untyped) -> Untyped:
        """Return ``x[idx]``. In addition to plain ``__getitem__``, this allows ensuring
        that the output is either a copy or a view; it also allows passing
        keyword arguments to the backend.
        """
        if kwargs.get("copy") is False:
            if is_array_api_obj(self.idx):
                # Boolean index. Note that the array API spec
                # https://data-apis.org/array-api/latest/API_specification/indexing.html
                # does not allow for list, tuple, and tuples of slices plus one or more
                # one-dimensional array indices, although many backends support them.
                # So this check will encounter a lot of false negatives in real life,
                # which can be caught by testing the user code vs. array-api-strict.
                msg = "get() with an array index always returns a copy"
                raise ValueError(msg)
            if is_dask_array(self.x):
                msg = "get() on Dask arrays always returns a copy"
                raise ValueError(msg)

        res, x = self._common("get", _is_update=False, **kwargs)
        if res is not None:
            return res
        assert x is not None
        return x[self.idx]

    def set(self, y: Array, /, **kwargs: Untyped) -> Array:
        """Apply ``x[idx] = y`` and return the update array"""
        res, x = self._common("set", y, **kwargs)
        if res is not None:
            return res
        assert x is not None
        x[self.idx] = y
        return x

    def _iop(
        self,
        at_op: Literal[
            "set", "add", "subtract", "multiply", "divide", "power", "min", "max"
        ],
        elwise_op: Callable[[Array, Array], Array],
        y: Array,
        /,
        **kwargs: Untyped,
    ) -> Array:
        """x[idx] += y or equivalent in-place operation on a subset of x

        which is the same as saying
            x[idx] = x[idx] + y
        Note that this is not the same as
            operator.iadd(x[idx], y)
        Consider for example when x is a numpy array and idx is a fancy index, which
        triggers a deep copy on __getitem__.
        """
        res, x = self._common(at_op, y, **kwargs)
        if res is not None:
            return res
        assert x is not None
        x[self.idx] = elwise_op(x[self.idx], y)
        return x

    def add(self, y: Array, /, **kwargs: Untyped) -> Array:
        """Apply ``x[idx] += y`` and return the updated array"""
        return self._iop("add", operator.add, y, **kwargs)

    def subtract(self, y: Array, /, **kwargs: Untyped) -> Array:
        """Apply ``x[idx] -= y`` and return the updated array"""
        return self._iop("subtract", operator.sub, y, **kwargs)

    def multiply(self, y: Array, /, **kwargs: Untyped) -> Array:
        """Apply ``x[idx] *= y`` and return the updated array"""
        return self._iop("multiply", operator.mul, y, **kwargs)

    def divide(self, y: Array, /, **kwargs: Untyped) -> Array:
        """Apply ``x[idx] /= y`` and return the updated array"""
        return self._iop("divide", operator.truediv, y, **kwargs)

    def power(self, y: Array, /, **kwargs: Untyped) -> Array:
        """Apply ``x[idx] **= y`` and return the updated array"""
        return self._iop("power", operator.pow, y, **kwargs)

    def min(self, y: Array, /, **kwargs: Untyped) -> Array:
        """Apply ``x[idx] = minimum(x[idx], y)`` and return the updated array"""
        xp = array_namespace(self.x)
        y = xp.asarray(y)
        return self._iop("min", xp.minimum, y, **kwargs)

    def max(self, y: Array, /, **kwargs: Untyped) -> Array:
        """Apply ``x[idx] = maximum(x[idx], y)`` and return the updated array"""
        xp = array_namespace(self.x)
        y = xp.asarray(y)
        return self._iop("max", xp.maximum, y, **kwargs)
