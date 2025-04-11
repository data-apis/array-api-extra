"""Backends with which array-api-extra interacts in delegation and testing."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from types import ModuleType

from ._utils import _compat

__all__ = ["Backend"]


class Backend(Enum):  # numpydoc ignore=PR01,PR02  # type: ignore[no-subclass-any]
    """
    All array library backends explicitly tested by array-api-extra.

    Parameters
    ----------
    value : str
        Name of the backend's module.
    is_namespace : Callable[[ModuleType], bool]
        Function to check whether an input module is the array namespace
        corresponding to the backend.
    """

    # Use :<tag> to prevent Enum from deduplicating items with the same value
    ARRAY_API_STRICT = "array_api_strict", _compat.is_array_api_strict_namespace
    ARRAY_API_STRICTEST = (
        "array_api_strict:strictest",
        _compat.is_array_api_strict_namespace,
    )
    NUMPY = "numpy", _compat.is_numpy_namespace
    NUMPY_READONLY = "numpy:readonly", _compat.is_numpy_namespace
    CUPY = "cupy", _compat.is_cupy_namespace
    TORCH = "torch", _compat.is_torch_namespace
    TORCH_GPU = "torch:gpu", _compat.is_torch_namespace
    DASK = "dask.array", _compat.is_dask_namespace
    SPARSE = "sparse", _compat.is_pydata_sparse_namespace
    JAX = "jax.numpy", _compat.is_jax_namespace
    JAX_GPU = "jax.numpy:gpu", _compat.is_jax_namespace

    def __new__(
        cls, value: str, _is_namespace: Callable[[ModuleType], bool]
    ):  # numpydoc ignore=GL08
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(
        self,
        value: str,  # noqa: ARG002  # pylint: disable=unused-argument
        is_namespace: Callable[[ModuleType], bool],
    ):  # numpydoc ignore=GL08
        self.is_namespace = is_namespace

    def __str__(self) -> str:  # type: ignore[explicit-override]  # pyright: ignore[reportImplicitOverride]  # numpydoc ignore=RT01
        """Pretty-print parameterized test names."""
        return (
            self.name.lower().replace("_gpu", ":gpu").replace("_readonly", ":readonly")
        )

    @property
    def modname(self) -> str:  # numpydoc ignore=RT01
        """Module name to be imported."""
        return self.value.split(":")[0]

    def like(self, *others: Backend) -> bool:  # numpydoc ignore=PR01,RT01
        """Check if this backend uses the same module as others."""
        return any(self.modname == other.modname for other in others)
