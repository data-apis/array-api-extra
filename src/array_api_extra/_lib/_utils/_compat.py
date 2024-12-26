"""Acquire helpers from array-api-compat."""
# Allow packages that vendor both `array-api-extra` and
# `array-api-compat` to override the import location

try:
    from ...._array_api_compat_vendor import (  # pyright: ignore[reportMissingImports]
        array_namespace,  # pyright: ignore[reportUnknownVariableType]
        device,  # pyright: ignore[reportUnknownVariableType]
        is_cupy_namespace,  # pyright: ignore[reportUnknownVariableType]
        is_jax_namespace,  # pyright: ignore[reportUnknownVariableType]
        is_numpy_namespace,  # pyright: ignore[reportUnknownVariableType]
        is_torch_namespace,  # pyright: ignore[reportUnknownVariableType]
    )
except ImportError:
    from array_api_compat import (  # pyright: ignore[reportMissingTypeStubs]
        array_namespace,  # pyright: ignore[reportUnknownVariableType]
        device,
        is_cupy_namespace,  # pyright: ignore[reportUnknownVariableType]
        is_jax_namespace,  # pyright: ignore[reportUnknownVariableType]
        is_numpy_namespace,  # pyright: ignore[reportUnknownVariableType]
        is_torch_namespace,  # pyright: ignore[reportUnknownVariableType]
    )

__all__ = [
    "array_namespace",
    "device",
    "is_cupy_namespace",
    "is_jax_namespace",
    "is_numpy_namespace",
    "is_torch_namespace",
]
