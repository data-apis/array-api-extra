from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Array, ModuleType

__all__ = ["atleast_nd"]


def atleast_nd(x: Array, *, ndim: int, xp: ModuleType) -> Array:
    """
    Recursively expand the dimension of an array to have at least `ndim`.

    Parameters
    ----------
    x: array
        An array.

    ndim: int
        The minimum number of dimensions for the result.

    xp: array_namespace
        The array namespace for `x`.

    Returns
    -------
    res: array
        An array with ``res.ndim`` >= `ndim`.
        If ``x.ndim`` >= `ndim`, `x` is returned.
        If ``x.ndim`` < `ndim`, ``res.ndim`` will equal `ndim`.
    """
    x = xp.asarray(x)
    if x.ndim < ndim:
        x = xp.expand_dims(x, axis=0)
        x = atleast_nd(x, ndim=ndim, xp=xp)
    return x
