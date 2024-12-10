# Allow packages that vendor both `array-api-extra` and
# `array-api-compat` to override the import location
from __future__ import annotations

try:
    from ..._array_api_compat_vendor import (
        array_namespace,
        device,
    )
except ImportError:
    from array_api_compat import (
        array_namespace,
        device,
    )

__all__ = [
    "array_namespace",
    "device",
]
