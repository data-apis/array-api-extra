"""Delegation to existing implementations for Public API Functions."""

import functools
from enum import Enum
from types import ModuleType
from typing import final

from ._lib import _funcs
from ._lib._utils._compat import (
    array_namespace,
    is_cupy_namespace,
    is_jax_namespace,
    is_numpy_namespace,
    is_torch_namespace,
)
from ._lib._utils._typing import Array

__all__ = ["pad"]


@final
class IsNamespace(Enum):
    """Enum to access is_namespace functions as the backend."""

    # TODO: when Python 3.10 is dropped, use `enum.member`
    # https://stackoverflow.com/a/74302109
    CUPY = functools.partial(is_cupy_namespace)
    JAX = functools.partial(is_jax_namespace)
    NUMPY = functools.partial(is_numpy_namespace)
    TORCH = functools.partial(is_torch_namespace)

    def __call__(self, xp: ModuleType) -> bool:
        """
        Call the is_namespace function.

        Parameters
        ----------
        xp : array_namespace
            Array namespace to check.

        Returns
        -------
        bool
            ``True`` if xp matches the namespace, ``False`` otherwise.
        """
        return self.value(xp)


CUPY = IsNamespace.CUPY
JAX = IsNamespace.JAX
NUMPY = IsNamespace.NUMPY
TORCH = IsNamespace.TORCH


def _delegate(xp: ModuleType, *backends: IsNamespace) -> bool:
    """
    Check whether `xp` is one of the `backends` to delegate to.

    Parameters
    ----------
    xp : array_namespace
        Array namespace to check.
    *backends : IsNamespace
        Arbitrarily many backends (from the ``IsNamespace`` enum) to check.

    Returns
    -------
    bool
        ``True`` if `xp` matches one of the `backends`, ``False`` otherwise.
    """
    return any(is_namespace(xp) for is_namespace in backends)


def pad(
    x: Array,
    pad_width: int | tuple[int, int] | list[tuple[int, int]],
    mode: str = "constant",
    *,
    constant_values: bool | int | float | complex = 0,
    xp: ModuleType | None = None,
) -> Array:
    """
    Pad the input array.

    Parameters
    ----------
    x : array
        Input array.
    pad_width : int or tuple of ints or list of pairs of ints
        Pad the input array with this many elements from each side.
        If a list of tuples, ``[(before_0, after_0), ... (before_N, after_N)]``,
        each pair applies to the corresponding axis of ``x``.
        A single tuple, ``(before, after)``, is equivalent to a list of ``x.ndim``
        copies of this tuple.
    mode : str, optional
        Only "constant" mode is currently supported, which pads with
        the value passed to `constant_values`.
    constant_values : python scalar, optional
        Use this value to pad the input. Default is zero.
    xp : array_namespace, optional
        The standard-compatible namespace for `x`. Default: infer.

    Returns
    -------
    array
        The input array,
        padded with ``pad_width`` elements equal to ``constant_values``.
    """
    xp = array_namespace(x) if xp is None else xp

    if mode != "constant":
        msg = "Only `'constant'` mode is currently supported"
        raise NotImplementedError(msg)

    # https://github.com/pytorch/pytorch/blob/cf76c05b4dc629ac989d1fb8e789d4fac04a095a/torch/_numpy/_funcs_impl.py#L2045-L2056
    if _delegate(xp, TORCH):
        pad_width = xp.asarray(pad_width)
        pad_width = xp.broadcast_to(pad_width, (x.ndim, 2))
        pad_width = xp.flip(pad_width, axis=(0,)).flatten()
        return xp.nn.functional.pad(x, (pad_width,), value=constant_values)

    if _delegate(xp, NUMPY, JAX, CUPY):
        return xp.pad(x, pad_width, mode, constant_values=constant_values)

    return _funcs.pad(x, pad_width, constant_values=constant_values, xp=xp)
