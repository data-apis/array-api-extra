from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
import pytest

from array_api_extra import default_dtype
from array_api_extra import searchsorted as xpx_searchsorted
from array_api_extra._agnostic._searching import searchsorted as _funcs_searchsorted
from array_api_extra._lib._backends import Backend
from array_api_extra._lib._compat import (
    array_namespace,
    is_jax_namespace,
    is_torch_namespace,
)
from array_api_extra._lib._typing import Array, ArrayNamespace
from array_api_extra.testing import assert_equal, lazy_xp_function

lazy_xp_function(xpx_searchsorted)
lazy_xp_function(_funcs_searchsorted)


def _apply_over_batch(*argdefs: tuple[str, int]) -> Any:
    """
    Factory for decorator that applies a function over batched arguments.

    Copied (with light simplifications) from `scipy._lib._util`.

    Array arguments may have any number of core dimensions (typically 0,
    1, or 2) and any broadcastable batch shapes. There may be any
    number of array outputs of any number of dimensions. Assumptions
    right now - which are satisfied by all functions of interest in `linalg` -
    are that all array inputs are consecutive keyword or positional arguments,
    and that the wrapped function returns either a single array or a tuple of
    arrays. It's only as general as it needs to be right now - it can be extended.

    Parameters
    ----------
    *argdefs : tuple of (str, int)
        Definitions of array arguments: the keyword name of the argument, and
        the number of core dimensions.

    Example:
    --------
    `linalg.eig` accepts two matrices as the first two arguments `a` and `b`, where
    `b` is optional, and returns one array or a tuple of arrays, depending on the
    values of other positional or keyword arguments. To generate a wrapper that applies
    the function over batches of `a` and optionally `b` :

    >>> _apply_over_batch(('a', 2), ('b', 2))
    """
    names, ndims = list(zip(*argdefs, strict=True))
    n_arrays = len(names)

    def decorator(f: Any) -> Any:
        def wrapper(
            *args_tuple: Any,
            **kwargs: Any,
        ) -> Any:
            args = list(args_tuple)

            # Ensure all arrays in `arrays`, other arguments in `other_args`/`kwargs`
            arrays, other_args = args[:n_arrays], args[n_arrays:]
            arrays = cast(list[Array | None], arrays)
            for i, name in enumerate(names):
                if name in kwargs:
                    if i + 1 <= len(args):
                        message = (
                            f"{f.__name__}() got multiple values for argument `{name}`."
                        )
                        raise ValueError(message)
                    arrays.append(kwargs.pop(name))

            xp = array_namespace(*arrays)

            # Determine core and batch shapes
            batch_shapes = []
            core_shapes = []
            for i, (array, ndim) in enumerate(zip(arrays, ndims, strict=True)):
                array = None if array is None else xp.asarray(array)  # noqa: PLW2901
                shape = () if array is None else array.shape
                arrays[i] = array
                batch_shapes.append(shape[:-ndim] if ndim > 0 else shape)
                core_shapes.append(shape[-ndim:] if ndim > 0 else ())

            # Early exit if call is not batched
            if not any(batch_shapes):
                return f(*arrays, *other_args, **kwargs)

            # Determine broadcasted batch shape
            batch_shape = np.broadcast_shapes(*batch_shapes)  # Gives OK error message

            # Broadcast arrays to appropriate shape
            for i, (array, core_shape) in enumerate(
                zip(arrays, core_shapes, strict=True)
            ):
                if array is None:
                    continue
                arrays[i] = xp.broadcast_to(array, batch_shape + core_shape)

            # Main loop
            results = []
            for index in np.ndindex(batch_shape):
                result = f(
                    *(
                        (array[index] if array is not None else None)
                        for array in arrays
                    ),
                    *other_args,
                    **kwargs,
                )
                # Assume `result` is either a tuple or single array. This is easily
                # generalized by allowing the contributor to pass an `unpack_result`
                # callable to the decorator factory.
                result = (result,) if not isinstance(result, tuple) else result
                results.append(result)
            results = list(zip(*results, strict=True))

            # Reshape results
            for i, result in enumerate(results):
                result = xp.stack(result)  # noqa: PLW2901
                core_shape = result.shape[1:]
                results[i] = xp.reshape(result, batch_shape + core_shape)

            # Assume `result` should be a single array if there is only one element or
            # a `tuple` otherwise. This is easily generalized by allowing the
            # contributor to pass an `pack_result` callable to the decorator factory.
            return results[0] if len(results) == 1 else results

        return wrapper

    return decorator


@_apply_over_batch(("a", 1), ("v", 1))  # type: ignore[untyped-decorator]
def xp_searchsorted(
    a: Array,
    v: Array,
    side: Literal["left", "right"],
    xp: ArrayNamespace,
) -> Array:
    return xp.searchsorted(a, v, side=side)


@pytest.mark.skip_xp_backend(Backend.DASK, reason="no take_along_axis")
@pytest.mark.skip_xp_backend(Backend.SPARSE, reason="no searchsorted")
class TestSearchsorted:
    def test_input_validation(self, xp: ArrayNamespace):
        message = "`side` must be either 'left' or 'right'."
        with pytest.raises(ValueError, match=message):
            _ = xpx_searchsorted(xp.asarray([1, 2]), xp.asarray([1, 2]), side="center")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("ties", [False, True])
    @pytest.mark.parametrize(
        "shape", [0, 1, 2, 10, 11, 1000, 10001, (2, 0), (0, 2), (2, 10), (2, 3, 11)]
    )
    @pytest.mark.parametrize("nans_x", [False, True])
    @pytest.mark.parametrize("infs_x", [False, True])
    @pytest.mark.parametrize("searchsorted", [xpx_searchsorted, _funcs_searchsorted])
    def test_nd(
        self,
        side: Literal["left", "right"],
        ties: bool,
        shape: int | tuple[int],
        nans_x: bool,
        infs_x: bool,
        xp: ArrayNamespace,
        searchsorted: Callable[..., Array],
    ):
        if nans_x and is_jax_namespace(xp):
            pytest.xfail("https://github.com/jax-ml/jax/issues/39887")
        if nans_x and is_torch_namespace(xp) and searchsorted == xpx_searchsorted:
            pytest.skip("torch sorts NaNs differently")
        if isinstance(shape, tuple) and searchsorted == _funcs_searchsorted:
            message = (
                "Redundant; `xpx_searchsorted` delegates to "
                "`_funcs_searchsorted` for multidimensional input."
            )
            pytest.skip(message)
        rng = np.random.default_rng(945298725498274853)
        x = rng.integers(5, size=shape) if ties else rng.random(shape)
        # float32 is to accommodate JAX - nextafter with `float64` is too small?
        x = np.asarray(x, dtype=np.float32)  # type:ignore[assignment]
        xr = np.nextafter(x, np.inf)
        xl = np.nextafter(x, -np.inf)
        x_ = np.asarray([-np.inf, np.inf, np.nan])
        x_ = np.broadcast_to(x_, (*x.shape[:-1], 3))
        y = rng.permuted(np.concatenate((xl, x, xr, x_), axis=-1), axis=-1)
        if nans_x:
            mask = rng.random(shape) < 0.1
            x[mask] = np.nan
        if infs_x:
            mask = rng.random(shape) < 0.1
            x[mask] = -np.inf
            mask = rng.random(shape) > 0.9
            x[mask] = np.inf
        x = np.sort(x, axis=-1)  # type:ignore[assignment]
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        xp_default_int = default_dtype(xp, kind="integral")
        if x.size == 0 and x.ndim > 0 and x.shape[-1] != 0:
            ref = xp.empty((*x.shape[:-1], y.shape[-1]), dtype=xp_default_int)
        else:
            ref = xp_searchsorted(x, y, side=side, xp=np)
            ref = xp.asarray(ref, dtype=xp_default_int)
        x, y = xp.asarray(x.copy()), xp.asarray(y.copy())
        res = searchsorted(x, y, side=side, xp=xp)
        assert_equal(res, ref)
