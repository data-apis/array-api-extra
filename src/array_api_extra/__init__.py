from __future__ import annotations

from ._funcs import atleast_nd, cov, create_diagonal, expand_dims, kron, setdiff1d, sinc

__version__ = "0.2.1.dev0"

# pylint: disable=duplicate-code
__all__ = [
    "__version__",
    "atleast_nd",
    "cov",
    "create_diagonal",
    "expand_dims",
    "kron",
    "setdiff1d",
    "sinc",
]
