"""Backends with which array-api-extra interacts in delegation and testing."""

from collections.abc import Callable
from enum import Enum
from types import ModuleType
from typing import cast

from ._utils import _compat

__all__ = ["Backend"]


class Backend(Enum):  # numpydoc ignore=PR01,PR02  # type: ignore[no-subclass-any]
    """
    All array library backends explicitly tested by array-api-extra.

    Parameters
    ----------
    value : str
        String describing the backend.
    is_namespace : Callable[[ModuleType], bool]
        Function to check whether an input module is the array namespace
        corresponding to the backend.
    module_name : str
        Name of the backend's module.
    """

    ARRAY_API_STRICT = (
        "array_api_strict",
        _compat.is_array_api_strict_namespace,
        "array_api_strict",
    )
    NUMPY = "numpy", _compat.is_numpy_namespace, "numpy"
    NUMPY_READONLY = "numpy_readonly", _compat.is_numpy_namespace, "numpy"
    CUPY = "cupy", _compat.is_cupy_namespace, "cupy"
    TORCH = "torch", _compat.is_torch_namespace, "torch"
    DASK_ARRAY = "dask.array", _compat.is_dask_namespace, "dask.array"
    SPARSE = "sparse", _compat.is_pydata_sparse_namespace, "sparse"
    JAX_NUMPY = "jax.numpy", _compat.is_jax_namespace, "jax.numpy"

    def __new__(
        cls, value: str, _is_namespace: Callable[[ModuleType], bool], _module_name: str
    ):  # numpydoc ignore=GL08
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(
        self,
        value: str,  # noqa: ARG002  # pylint: disable=unused-argument
        is_namespace: Callable[[ModuleType], bool],
        module_name: str,
    ):  # numpydoc ignore=GL08
        self.is_namespace = is_namespace
        self.module_name = module_name

    def __str__(self) -> str:  # type: ignore[explicit-override]  # pyright: ignore[reportImplicitOverride]  # numpydoc ignore=RT01
        """Pretty-print parameterized test names."""
        return cast(str, self.value)
