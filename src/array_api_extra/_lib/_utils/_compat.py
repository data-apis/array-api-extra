"""Acquire helpers from array-api-compat."""
# Allow packages that vendor both `array-api-extra` and
# `array-api-compat` to override the import location

try:
    from ...._array_api_compat_vendor import (  # pyright: ignore[reportMissingImports]
        array_namespace,
        device,
        is_array_api_strict_namespace,
        is_cupy_namespace,
        is_dask_namespace,
        is_jax_array,
        is_jax_namespace,
        is_numpy_namespace,
        is_pydata_sparse_namespace,
        is_torch_namespace,
        is_writeable_array,
        size,
    )
except ImportError:
    from array_api_compat import (  # pyright: ignore[reportMissingTypeStubs]
        array_namespace,
        device,
        is_array_api_strict_namespace,
        is_cupy_namespace,
        is_dask_namespace,
        is_jax_array,
        is_jax_namespace,
        is_numpy_namespace,
        is_pydata_sparse_namespace,
        is_torch_namespace,
        is_writeable_array,
        size,
    )

__all__ = [
    "array_namespace",
    "device",
    "is_array_api_strict_namespace",
    "is_cupy_namespace",
    "is_dask_namespace",
    "is_jax_array",
    "is_jax_namespace",
    "is_numpy_namespace",
    "is_pydata_sparse_namespace",
    "is_torch_namespace",
    "is_writeable_array",
    "size",
]
