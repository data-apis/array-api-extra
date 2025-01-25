"""Public API Functions."""

# https://github.com/scikit-learn/scikit-learn/pull/27910#issuecomment-2568023972
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import partial, wraps
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast, overload

from ._utils._compat import (
    array_namespace,
    is_array_api_obj,
    is_dask_namespace,
    is_jax_array,
    is_jax_namespace,
)
from ._utils._typing import Array, DType

if TYPE_CHECKING:
    # TODO move outside TYPE_CHECKING
    # depends on scikit-learn abandoning Python 3.9
    # https://github.com/scikit-learn/scikit-learn/pull/27910#issuecomment-2568023972
    from typing import ParamSpec, TypeAlias

    import numpy as np
    from numpy.typing import ArrayLike

    NumPyObject: TypeAlias = np.ndarray[Any, Any] | np.generic  # type: ignore[no-any-explicit]
    P = ParamSpec("P")
else:
    # Sphinx hacks
    NumPyObject = Any

    class P:  # pylint: disable=missing-class-docstring
        args: tuple
        kwargs: dict


@overload
def lazy_apply(  # type: ignore[valid-type]
    func: Callable[P, ArrayLike],
    *args: Array,
    shape: tuple[int | None, ...] | None = None,
    dtype: DType | None = None,
    as_numpy: bool = False,
    xp: ModuleType | None = None,
    **kwargs: P.kwargs,  # pyright: ignore[reportGeneralTypeIssues]
) -> Array: ...  # numpydoc ignore=GL08


@overload
def lazy_apply(  # type: ignore[valid-type]
    func: Callable[P, Sequence[ArrayLike]],
    *args: Array,
    shape: Sequence[tuple[int | None, ...]],
    dtype: Sequence[DType] | None = None,
    as_numpy: bool = False,
    xp: ModuleType | None = None,
    **kwargs: P.kwargs,  # pyright: ignore[reportGeneralTypeIssues]
) -> tuple[Array, ...]: ...  # numpydoc ignore=GL08


def lazy_apply(  # type: ignore[valid-type]  # numpydoc ignore=GL07,SA04
    func: Callable[P, Array | Sequence[ArrayLike]],
    *args: Array,
    shape: tuple[int | None, ...] | Sequence[tuple[int | None, ...]] | None = None,
    dtype: DType | Sequence[DType] | None = None,
    as_numpy: bool = False,
    xp: ModuleType | None = None,
    **kwargs: P.kwargs,  # pyright: ignore[reportGeneralTypeIssues]
) -> Array | tuple[Array, ...]:
    """
    Lazily apply an eager function.

    If the backend of the input arrays is lazy, e.g. Dask or jitted JAX, the execution
    of the function is delayed until the graph is materialized; if it's eager, the
    function is executed immediately.

    Parameters
    ----------
    func : callable
        The function to apply.

        It must accept one or more array API compliant arrays as positional arguments.
        If `as_numpy=True`, inputs are converted to NumPy before they are passed to
        `func`.
        It must return either a single array-like or a sequence of array-likes.

        `func` must be a pure function, i.e. without side effects, as depending on the
        backend it may be executed more than once.
    *args : Array
        One or more Array API compliant arrays.

        If `as_numpy=True`, you need to be able to apply :func:`numpy.asarray` to them
        to convert them to numpy; read notes below about specific backends.
    shape : tuple[int | None, ...] | Sequence[tuple[int, ...]], optional
        Output shape or sequence of output shapes, one for each output of `func`.
        Default: assume single output and broadcast shapes of the input arrays.
    dtype : DType | Sequence[DType], optional
        Output dtype or sequence of output dtypes, one for each output of `func`.
        dtype(s) must belong to the same array namespace as the input arrays.
        Default: infer the result type(s) from the input arrays.
    as_numpy : bool, optional
        If True, convert the input arrays to NumPy before passing them to `func`.
        This is particularly useful to make numpy-only functions, e.g. written in Cython
        or Numba, work transparently API arrays.
        Default: False.
    xp : array_namespace, optional
        The standard-compatible namespace for `args`. Default: infer.
    **kwargs : Any, optional
        Additional keyword arguments to pass verbatim to `func`.
        Any array objects in them will be converted to numpy when ``as_numpy=True``.

    Returns
    -------
    Array | tuple[Array, ...]
        The result(s) of `func` applied to the input arrays, wrapped in the same
        array namespace as the inputs.
        If shape is omitted or a `tuple[int | None, ...]`, this is a single array.
        Otherwise, it's a tuple of arrays.

    Notes
    -----
    JAX
        This allows applying eager functions to jitted JAX arrays, which are lazy.
        The function won't be applied until the JAX array is materialized.
        When running inside `jax.jit`, `shape` must be fully known, i.e. it cannot
        contain any `None` elements.

        Using this with `as_numpy=False` is particularly useful to apply non-jittable
        JAX functions to arrays on GPU devices.
        If `as_numpy=True`, the :doc:`jax:transfer_guard` may prevent arrays on a GPU
        device from being transferred back to CPU. This is treated as an implicit
        transfer.

    PyTorch, CuPy
        If `as_numpy=True`, these backends raise by default if you attempt to convert
        arrays on a GPU device to NumPy.

    Sparse
        If `as_numpy=True`, by default sparse prevents implicit densification through
        :func:`numpy.asarray`. `This safety mechanism can be disabled
        <https://sparse.pydata.org/en/stable/operations.html#package-configuration>`_.

    Dask
        This allows applying eager functions to dask arrays.
        The dask graph won't be computed.

        `lazy_apply` doesn't know if `func` reduces along any axes; also, shape
        changes are non-trivial in chunked Dask arrays. For these reasons, all inputs
        will be rechunked into a single chunk.

        .. warning::

           The whole operation needs to fit in memory all at once on a single worker.

        The outputs will also be returned as a single chunk and you should consider
        rechunking them into smaller chunks afterwards.

        If you want to distribute the calculation across multiple workers, you
        should use :func:`dask.array.map_blocks`, :func:`dask.array.map_overlap`,
        :func:`dask.array.blockwise`, or a native Dask wrapper instead of
        `lazy_apply`.

    Dask wrapping around other backends
        If `as_numpy=False`, `func` will receive in input eager arrays of the meta
        namespace, as defined by the `._meta` attribute of the input Dask arrays.
        The outputs of `func` will be wrapped by the meta namespace, and then wrapped
        again by Dask.

    Raises
    ------
    jax.errors.TracerArrayConversionError
        When `xp=jax.numpy`, `shape` is unknown (it contains None on one or more axes)
        and this function was called inside `jax.jit`.
    RuntimeError
        When `xp=sparse` and auto-densification is disabled.
    Exception (backend-specific)
        When the backend disallows implicit device to host transfers and the input
        arrays are on a device, e.g. on GPU.

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
    shapes: list[tuple[int | None, ...]]
    dtypes: list[DType]
    multi_output = False

    if shape is None:
        shapes = [xp.broadcast_shapes(*(arg.shape for arg in args))]
    elif isinstance(shape, tuple) and all(isinstance(s, int | None) for s in shape):
        shapes = [shape]  # pyright: ignore[reportAssignmentType]
    else:
        shapes = list(shape)  # type: ignore[arg-type]  # pyright: ignore[reportAssignmentType]
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
        import dask

        metas = [arg._meta for arg in args if hasattr(arg, "_meta")]  # pylint: disable=protected-access
        meta_xp = array_namespace(*metas)

        wrapped = dask.delayed(  # type: ignore[attr-defined]  # pyright: ignore[reportPrivateImportUsage]
            _lazy_apply_wrapper(func, as_numpy, multi_output, meta_xp),
            pure=True,
        )
        # This finalizes each arg, which is the same as arg.rechunk(-1).
        # Please read docstring above for why we're not using
        # dask.array.map_blocks or dask.array.blockwise!
        delayed_out = wrapped(*args, **kwargs)

        out = tuple(
            xp.from_delayed(
                delayed_out[i],  # pyright: ignore[reportIndexIssue]
                # Dask's unknown shapes diverge from the Array API specification
                shape=tuple(math.nan if s is None else s for s in shape),
                dtype=dtype,
                meta=metas[0],
            )
            for i, (shape, dtype) in enumerate(zip(shapes, dtypes, strict=True))
        )

    elif is_jax_namespace(xp):
        # If we're inside jax.jit, we can't eagerly convert
        # the JAX tracer objects to numpy.
        # Instead, we delay calling wrapped, which will receive
        # as arguments and will return JAX eager arrays.

        import jax

        # Shield eager kwargs from being coerced into JAX arrays.
        # jax.pure_callback calls jax.jit under the hood, but without the chance of
        # passing static_argnames / static_argnums.
        lazy_kwargs = {}
        eager_kwargs = {}
        for k, v in kwargs.items():
            if _contains_jax_arrays(v):
                lazy_kwargs[k] = v
            else:
                eager_kwargs[k] = v

        wrapped = _lazy_apply_wrapper(
            partial(func, **eager_kwargs), as_numpy, multi_output, xp
        )

        if any(s is None for shape in shapes for s in shape):
            # Unknown output shape. Won't work with jax.jit, but it
            # can work with eager jax.
            # Raises jax.errors.TracerArrayConversionError if we're inside jax.jit.
            out = wrapped(*args, **lazy_kwargs)

        else:
            # suppress unused-ignore to run mypy in -e lint as well as -e dev
            out = cast(  # type: ignore[bad-cast,unused-ignore]
                tuple[Array, ...],
                jax.pure_callback(
                    wrapped,
                    tuple(
                        jax.ShapeDtypeStruct(shape, dtype)  # pyright: ignore[reportUnknownArgumentType]
                        for shape, dtype in zip(shapes, dtypes, strict=True)
                    ),
                    *args,
                    **lazy_kwargs,
                ),
            )

    else:
        # Eager backends
        wrapped = _lazy_apply_wrapper(func, as_numpy, multi_output, xp)
        out = wrapped(*args, **kwargs)

    return out if multi_output else out[0]


def _contains_jax_arrays(x: object) -> bool:  # numpydoc ignore=PR01,RT01
    """
    Test if x is a JAX array or a nested collection with any JAX arrays in it.
    """
    if is_jax_array(x):
        return True
    if isinstance(x, list | tuple):
        return any(_contains_jax_arrays(i) for i in x)  # pyright: ignore[reportUnknownArgumentType]
    if isinstance(x, dict):
        return any(_contains_jax_arrays(i) for i in x.values())  # pyright: ignore[reportUnknownArgumentType]
    return False


def _as_numpy(x: object) -> Any:  # type: ignore[no-any-explicit] # numpydoc ignore=PR01,RT01
    """Recursively convert Array API objects in x to NumPy."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    if is_array_api_obj(x):
        return np.asarray(x)
    if isinstance(x, list) or type(x) is tuple:  # pylint: disable=unidiomatic-typecheck
        return type(x)(_as_numpy(i) for i in x)  # pyright: ignore[reportUnknownArgumentType]
    if isinstance(x, tuple):  # namedtuple
        return type(x)(*(_as_numpy(i) for i in x))  # pyright: ignore[reportUnknownArgumentType]
    if isinstance(x, dict):
        return {k: _as_numpy(v) for k, v in x.items()}  # pyright: ignore[reportUnknownArgumentType]
    return x


def _lazy_apply_wrapper(  # type: ignore[no-any-explicit]  # numpydoc ignore=PR01,RT01
    func: Callable[..., ArrayLike | Sequence[ArrayLike]],
    as_numpy: bool,
    multi_output: bool,
    xp: ModuleType,
) -> Callable[..., tuple[Array, ...]]:
    """
    Helper of `lazy_apply`.

    Given a function that accepts one or more arrays as positional arguments and returns
    a single array-like or a sequence of array-likes, return a function that accepts the
    same number of Array API arrays and always returns a tuple of Array API array.

    Any keyword arguments are passed through verbatim to the wrapped function.
    """

    # On Dask, @wraps causes the graph key to contain the wrapped function's name
    @wraps(func)
    def wrapper(  # type: ignore[no-any-decorated,no-any-explicit]
        *args: Array, **kwargs: Any
    ) -> tuple[Array, ...]:  # numpydoc ignore=GL08
        if as_numpy:
            args = _as_numpy(args)
            kwargs = _as_numpy(kwargs)
        out = func(*args, **kwargs)

        if multi_output:
            assert isinstance(out, Sequence)
            return tuple(xp.asarray(o) for o in out)
        return (xp.asarray(out),)

    return wrapper
