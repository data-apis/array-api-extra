"""Extra array functions built on top of the array API standard."""

from ._funcs import (
    atleast_nd,
    cov,
    create_diagonal,
    expand_dims,
    kron,
    pad,
    setdiff1d,
    sinc,
)

__version__ = "0.4.1.dev0"

# pylint: disable=duplicate-code
__all__ = [
    "__version__",
    "atleast_nd",
    "cov",
    "create_diagonal",
    "expand_dims",
    "kron",
    "pad",
    "setdiff1d",
    "sinc",
]
