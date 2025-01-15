"""Delegation to existing implementations for Public API Functions."""

from types import ModuleType
from typing import Literal

from ._lib import Backend, _funcs
from ._lib._utils._compat import array_namespace
from ._lib._utils._typing import Array

__all__ = ["pad"]


def _delegate(xp: ModuleType, *backends: Backend) -> bool:
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
    return any(backend.is_namespace(xp) for backend in backends)


def pad(
    x: Array,
    pad_width: int | tuple[int, int] | list[tuple[int, int]],
    mode: Literal["constant"] = "constant",
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
    if _delegate(xp, Backend.TORCH):
        pad_width = xp.asarray(pad_width)
        pad_width = xp.broadcast_to(pad_width, (x.ndim, 2))
        pad_width = xp.flip(pad_width, axis=(0,)).flatten()
        return xp.nn.functional.pad(x, tuple(pad_width), value=constant_values)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

    if _delegate(xp, Backend.NUMPY, Backend.JAX_NUMPY, Backend.CUPY):
        return xp.pad(x, pad_width, mode, constant_values=constant_values)

    return _funcs.pad(x, pad_width, constant_values=constant_values, xp=xp)
