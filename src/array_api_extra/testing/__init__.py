"""Public testing utilities."""

from ._testing import (
    assert_close,
    assert_close_nulp,
    assert_equal,
    assert_less,
    lazy_xp_function,
    patch_lazy_xp_functions,
    yield_namespace_device_dtype_combinations,
    yield_namespaces,
)

__all__ = [
    "assert_close",
    "assert_close_nulp",
    "assert_equal",
    "assert_less",
    "lazy_xp_function",
    "patch_lazy_xp_functions",
    "yield_namespace_device_dtype_combinations",
    "yield_namespaces",
]
