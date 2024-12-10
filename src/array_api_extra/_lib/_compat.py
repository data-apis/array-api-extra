# Allow packages that vendor both `array-api-extra` and
# `array-api-compat` to override the import location
from __future__ import annotations

try:
    from ..._array_api_compat_vendor import (  # pyright: ignore[reportMissingImports]
        array_namespace,
        device,
        is_array_api_obj,
        is_dask_array,
        is_writeable_array,
    )
except ImportError:
    from array_api_compat import (  # pyright: ignore[reportMissingTypeStubs]
        array_namespace,
        device,
        is_array_api_obj,
        is_dask_array,
        is_writeable_array,
    )

__all__ = (
    "array_namespace",
    "device",
    "is_array_api_obj",
    "is_dask_array",
    "is_writeable_array",
)
