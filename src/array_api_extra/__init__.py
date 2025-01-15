"""Extra array functions built on top of the array API standard."""

from ._delegation import pad
from ._lib._funcs import (
    at,
    atleast_nd,
    cov,
    create_diagonal,
    expand_dims,
    kron,
    nunique,
    setdiff1d,
    sinc,
)

__version__ = "0.6.1.dev0"

# pylint: disable=duplicate-code
__all__ = [
    "__version__",
    "at",
    "atleast_nd",
    "cov",
    "create_diagonal",
    "expand_dims",
    "kron",
    "nunique",
    "pad",
    "setdiff1d",
    "sinc",
]
