"""Extra array functions built on top of the array API standard."""

from ._delegation import isclose, pad
from ._lib._at import at
from ._lib._funcs import (
    apply_where,
    atleast_nd,
    broadcast_shapes,
    cov,
    create_diagonal,
    expand_dims,
    kron,
    nunique,
    setdiff1d,
    sinc,
)

__version__ = "0.7.0.dev0"

# pylint: disable=duplicate-code
__all__ = [
    "__version__",
    "apply_where",
    "at",
    "atleast_nd",
    "broadcast_shapes",
    "cov",
    "create_diagonal",
    "expand_dims",
    "isclose",
    "kron",
    "nunique",
    "pad",
    "setdiff1d",
    "sinc",
]
