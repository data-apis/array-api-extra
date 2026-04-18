"""This file is a hook imported by `src/array_api_extra/_lib/_compat.py`."""
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false

from types import ModuleType
from typing import Any

from .array_api_compat import *  # noqa: F403 # pyright: ignore[reportAssignmentType]
from .array_api_compat import array_namespace as array_namespace_compat


# Let unit tests check with `is` that we are picking up the function from this module
# and not from the original array_api_compat module.
def array_namespace(*xs: Any | complex | None, **kwargs) -> ModuleType:  # type: ignore[no-redef] # numpydoc ignore=GL08
    return array_namespace_compat(*xs, **kwargs)  # pyright: ignore[reportUnknownArgumentType]
