from __future__ import annotations  # https://github.com/pylint-dev/pylint/pull/9990

import typing

if typing.TYPE_CHECKING:
    from ._typing import Array, ModuleType

from . import _compat

__all__ = ["in1d"]


def in1d(
    x1: Array,
    x2: Array,
    /,
    *,
    assume_unique: bool = False,
    invert: bool = False,
    xp: ModuleType,
) -> Array:
    """Checks whether each element of an array is also present in a
    second array.

    Returns a boolean array the same length as `x1` that is True
    where an element of `x1` is in `x2` and False otherwise.

    This function has been adapted using the original implementation
    present in numpy:
    https://github.com/numpy/numpy/blob/v1.26.0/numpy/lib/arraysetops.py#L524-L758
    """

    # This code is run to make the code significantly faster
    if x2.shape[0] < 10 * x1.shape[0] ** 0.145:
        if invert:
            mask = xp.ones(x1.shape[0], dtype=xp.bool, device=x1.device)
            for a in x2:
                mask &= x1 != a
        else:
            mask = xp.zeros(x1.shape[0], dtype=xp.bool, device=x1.device)
            for a in x2:
                mask |= x1 == a
        return mask

    rev_idx = xp.empty(0)  # placeholder
    if not assume_unique:
        x1, rev_idx = xp.unique_inverse(x1)
        x2 = xp.unique_values(x2)

    ar = xp.concat((x1, x2))
    device_ = _compat.device(ar)
    # We need this to be a stable sort.
    order = xp.argsort(ar, stable=True)
    reverse_order = xp.argsort(order, stable=True)
    sar = xp.take(ar, order, axis=0)
    if sar.size >= 1:
        bool_ar = sar[1:] != sar[:-1] if invert else sar[1:] == sar[:-1]
    else:
        bool_ar = xp.asarray([False]) if invert else xp.asarray([True])
    flag = xp.concat((bool_ar, xp.asarray([invert], device=device_)))
    ret = xp.take(flag, reverse_order, axis=0)

    if assume_unique:
        return ret[: x1.shape[0]]
    return xp.take(ret, rev_idx, axis=0)
