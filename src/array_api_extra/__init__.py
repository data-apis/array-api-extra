from __future__ import annotations  # https://github.com/pylint-dev/pylint/pull/9990

from ._funcs import (
    at,
    atleast_nd,
    cov,
    create_diagonal,
    expand_dims,
    kron,
    setdiff1d,
    sinc,
)

__version__ = "0.3.3.dev0"

# pylint: disable=duplicate-code
__all__ = [
    "__version__",
    "at",
    "atleast_nd",
    "cov",
    "create_diagonal",
    "expand_dims",
    "kron",
    "setdiff1d",
    "sinc",
]
