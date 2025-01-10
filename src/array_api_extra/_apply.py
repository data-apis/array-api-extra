"""Public API Functions."""

# https://github.com/scikit-learn/scikit-learn/pull/27910#issuecomment-2568023972
from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import wraps
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast, overload

from ._lib._compat import (
    array_namespace,
    is_dask_namespace,
    is_jax_namespace,
)
from ._lib._typing import Array, DType

if TYPE_CHECKING:
    # https://github.com/scikit-learn/scikit-learn/pull/27910#issuecomment-2568023972
    from typing import ParamSpec, TypeAlias

    import numpy as np

    NumPyObject: TypeAlias = np.ndarray[Any, Any] | np.generic  # type: ignore[no-any-explicit]
    P = ParamSpec("P")
else:
    # Sphinx hacks
    NumPyObject = Any

    class P:  # pylint: disable=missing-class-docstring
        args: tuple
        kwargs: dict


@overload
def apply_numpy_func(  # type: ignore[valid-type]
    func: Callable[P, NumPyObject],
    *args: Array,
    shape: tuple[int, ...] | None = None,
    dtype: DType | None = None,
    xp: ModuleType | None = None,
    **kwargs: P.kwargs,  # pyright: ignore[reportGeneralTypeIssues]
) -> Array: ...  # numpydoc ignore=GL08


@overload
def apply_numpy_func(  # type: ignore[valid-type]
    func: Callable[P, Sequence[NumPyObject]],
    *args: Array,
    shape: Sequence[tuple[int, ...]],
    dtype: Sequence[DType] | None = None,
    xp: ModuleType | None = None,
    **kwargs: P.kwargs,  # pyright: ignore[reportGeneralTypeIssues]
) -> tuple[Array, ...]: ...  # numpydoc ignore=GL08


def apply_numpy_func(  # type: ignore[valid-type]  # numpydoc ignore=GL07,SA04
    func: Callable[P, NumPyObject | Sequence[NumPyObject]],
    *args: Array,
    shape: tuple[int, ...] | Sequence[tuple[int, ...]] | None = None,
    dtype: DType | Sequence[DType] | None = None,
    xp: ModuleType | None = None,
    **kwargs: P.kwargs,  # pyright: ignore[reportGeneralTypeIssues]
) -> Array | tuple[Array, ...]:
    """
    Apply a function that operates on NumPy arrays to Array API compliant arrays.

    Parameters
    ----------
    func : callable
        The function to apply. It must accept one or more NumPy arrays or generics as
        positional arguments and return either a single NumPy array or generic, or a
        tuple or list thereof.

        It must be a pure function, i.e. without side effects such as disk output,
        as depending on the backend it may be executed more than once.
    *args : Array
        One or more Array API compliant arrays. You need to be able to apply
        :func:`numpy.asarray` to them to convert them to numpy; read notes below about
        specific backends.
    shape : tuple[int, ...] | Sequence[tuple[int, ...]], optional
        Output shape or sequence of output shapes, one for each output of `func`.
        Default: assume single output and broadcast shapes of the input arrays.
    dtype : DType | Sequence[DType], optional
        Output dtype or sequence of output dtypes, one for each output of `func`.
        dtype(s) must belong to the same array namespace as the input arrays.
        Default: infer the result type(s) from the input arrays.
    xp : array_namespace, optional
        The standard-compatible namespace for `args`. Default: infer.
    **kwargs : Any, optional
        Additional keyword arguments to pass verbatim to `func`.
        Any array objects in them won't be converted to NumPy.

    Returns
    -------
    Array | tuple[Array, ...]
        The result(s) of `func` applied to the input arrays, wrapped in the same
        array namespace as the inputs.
        If shape is omitted or a `tuple[int, ...]`, this is a single array.
        Otherwise, it's a tuple of arrays.

    Notes
    -----
    JAX
        This allows applying eager functions to jitted JAX arrays, which are lazy.
        The function won't be applied until the JAX array is materialized.

        The :doc:`jax:transfer_guard` may prevent arrays on a GPU device from being
        transferred back to CPU. This is treated as an implicit transfer.

    PyTorch, CuPy
        These backends raise by default if you attempt to convert arrays on a GPU device
        to NumPy.

    Sparse
        By default, sparse prevents implicit densification through
        :func:`numpy.asarray`. `This safety mechanism can be disabled
        <https://sparse.pydata.org/en/stable/operations.html#package-configuration>`_.

    Dask
        This allows applying eager functions to dask arrays.
        The dask graph won't be computed.

        `apply_numpy_func` doesn't know if `func` reduces along any axes; also, shape
        changes are non-trivial in chunked Dask arrays. For these reasons, all inputs
        will be rechunked into a single chunk.

        .. warning::

           The whole operation needs to fit in memory all at once on a single worker.

        The outputs will also be returned as a single chunk and you should consider
        rechunking them into smaller chunks afterwards.

        If you want to distribute the calculation across multiple workers, you
        should use :func:`dask.array.map_blocks`, :func:`dask.array.map_overlap`,
        :func:`dask.array.blockwise`, or a native Dask wrapper instead of
        `apply_numpy_func`.

    See Also
    --------
    jax.transfer_guard
    jax.pure_callback
    dask.array.map_blocks
    dask.array.map_overlap
    dask.array.blockwise
    """
    if xp is None:
        xp = array_namespace(*args)

    # Normalize and validate shape and dtype
    multi_output = False
    if shape is None:
        shapes = [xp.broadcast_shapes(*(arg.shape for arg in args))]
    elif isinstance(shape, tuple) and all(isinstance(s, int) for s in shape):
        shapes = [shape]
    else:
        shapes = list(shape)
        multi_output = True

    if dtype is None:
        dtypes = [xp.result_type(*args)] * len(shapes)
    elif multi_output:
        if not isinstance(dtype, Sequence):
            msg = "Got sequence of shapes but only one dtype"
            raise TypeError(msg)
        dtypes = list(dtype)  # pyright: ignore[reportUnknownArgumentType]
    else:
        if isinstance(dtype, Sequence):
            msg = "Got single shape but multiple dtypes"
            raise TypeError(msg)
        dtypes = [dtype]

    if len(shapes) != len(dtypes):
        msg = f"Got {len(shapes)} shapes and {len(dtypes)} dtypes"
        raise ValueError(msg)
    if len(shapes) == 0:
        msg = "func must return one or more output arrays"
        raise ValueError(msg)
    del shape
    del dtype

    # Backend-specific branches
    if is_dask_namespace(xp):
        import dask  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel,import-error  # pyright: ignore[reportMissingImports]

        metas = [arg._meta for arg in args if hasattr(arg, "_meta")]  # pylint: disable=protected-access
        meta_xp = array_namespace(*metas)

        wrapped = dask.delayed(_npfunc_wrapper(func, multi_output, meta_xp), pure=True)
        # This finalizes each arg, which is the same as arg.rechunk(-1)
        # Please read docstring above for why we're not using
        # dask.array.map_blocks or dask.array.blockwise!
        delayed_out = wrapped(*args, **kwargs)

        out = tuple(
            xp.from_delayed(delayed_out[i], shape=shape, dtype=dtype, meta=metas[0])
            for i, (shape, dtype) in enumerate(zip(shapes, dtypes, strict=True))
        )

    elif is_jax_namespace(xp):
        # If we're inside jax.jit, we can't eagerly convert
        # the JAX tracer objects to numpy.
        # Instead, we delay calling wrapped, which will receive
        # as arguments and will return JAX eager arrays.

        import jax  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel,import-error  # pyright: ignore[reportMissingImports]

        wrapped = _npfunc_wrapper(func, multi_output, xp)
        out = cast(
            tuple[Array, ...],
            jax.pure_callback(
                wrapped,
                tuple(
                    jax.ShapeDtypeStruct(s, dt)  # pyright: ignore[reportUnknownArgumentType]
                    for s, dt in zip(shapes, dtypes, strict=True)
                ),
                *args,
                **kwargs,
            ),
        )

    else:
        # Eager backends
        wrapped = _npfunc_wrapper(func, multi_output, xp)
        out = wrapped(*args, **kwargs)

        # Output validation
        if len(out) != len(shapes):
            msg = f"func was declared to return {len(shapes)} outputs, got {len(out)}"
            raise ValueError(msg)
        for out_i, shape_i, dtype_i in zip(out, shapes, dtypes, strict=True):
            if out_i.shape != shape_i:
                msg = f"expected shape {shape_i}, got {out_i.shape}"
                raise ValueError(msg)
            if not xp.isdtype(out_i.dtype, dtype_i):
                msg = f"expected dtype {dtype_i}, got {out_i.dtype}"
                raise ValueError(msg)

    return out if multi_output else out[0]


def _npfunc_wrapper(  # type: ignore[no-any-explicit]  # numpydoc ignore=PR01,RT01
    func: Callable[..., NumPyObject | Sequence[NumPyObject]],
    multi_output: bool,
    xp: ModuleType,
) -> Callable[..., tuple[Array, ...]]:
    """
    Helper of `apply_numpy_func`.

    Given a function that accepts one or more numpy arrays as positional arguments and
    returns a single numpy array or a sequence of numpy arrays, return a function that
    accepts the same number of Array API arrays and always returns a tuple of Array API
    array.

    Any keyword arguments are passed through verbatim to the wrapped function.

    Raise if np.asarray raises on any input. This typically happens if the input is lazy
    and has a guard against being implicitly turned into a NumPy array (e.g.
    densification for sparse arrays, device->host transfer for cupy and torch arrays).
    """

    # On Dask, @wraps causes the graph key to contain the wrapped function's name
    @wraps(func)
    def wrapper(  # type: ignore[no-any-decorated,no-any-explicit]
        *args: Array, **kwargs: Any
    ) -> tuple[Array, ...]:  # numpydoc ignore=GL08
        import numpy as np  # pylint: disable=import-outside-toplevel

        args = tuple(np.asarray(arg) for arg in args)
        out = func(*args, **kwargs)

        # Stay relaxed on output validation, e.g. in case func returns a
        # Python scalar instead of a np.generic
        if multi_output:
            if not isinstance(out, Sequence) or isinstance(out, np.ndarray):
                msg = "Expected multiple outputs, got a single one"
                raise ValueError(msg)
            outs = out
        else:
            outs = [cast("NumPyObject", out)]

        return tuple(xp.asarray(o) for o in outs)

    return wrapper
