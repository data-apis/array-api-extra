# Allow packages that vendor both `array-api-extra` and
# `array-api-compat` to override the import location

try:
    from ..._array_api_compat_vendor import (  # pyright: ignore[reportMissingImports]
        array_namespace,  # pyright: ignore[reportUnknownVariableType]
        device,  # pyright: ignore[reportUnknownVariableType]
    )
except ImportError:
    from array_api_compat import (  # pyright: ignore[reportMissingTypeStubs]
        array_namespace,  # pyright: ignore[reportUnknownVariableType]
        device,
    )

__all__ = [
    "array_namespace",
    "device",
]
