# This file is a hook imported by src/array_api_extra/_lib/_compat.py
from __future__ import annotations

from .array_api_compat import *  # noqa: F403
from .array_api_compat import array_namespace as array_namespace_compat


# Let unit tests check with `is` that we are picking up the function from this module
# and not from the original array_api_compat module.
def array_namespace(*xs, **kwargs):
    return array_namespace_compat(*xs, **kwargs)
