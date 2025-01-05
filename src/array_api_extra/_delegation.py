"""Delegation to existing implementations for Public API Functions."""

from types import ModuleType

from ._lib import _funcs
from ._lib._utils._compat import (
    array_namespace,
    is_cupy_namespace,
    is_jax_namespace,
    is_numpy_namespace,
    is_torch_namespace,
)
from ._lib._utils._typing import Array


def pad(
    x: Array,
    pad_width: int,
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
    pad_width : int
        Pad the input array with this many elements from each side.
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
    if is_torch_namespace(xp):
        pad_width = xp.asarray(pad_width)
        pad_width = xp.broadcast_to(pad_width, (x.ndim, 2))
        pad_width = xp.flip(pad_width, axis=(0,)).flatten()
        return xp.nn.functional.pad(x, (pad_width,), value=constant_values)

    if is_numpy_namespace(xp) or is_jax_namespace(xp) or is_cupy_namespace(xp):
        return xp.pad(x, pad_width, mode, constant_values=constant_values)

    return _funcs.pad(x, pad_width, constant_values=constant_values, xp=xp)
