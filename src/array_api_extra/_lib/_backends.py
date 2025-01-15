"""Code specifying libraries array-api-extra interacts with."""

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
    library_name : str
        Name of the array library of the backend.
    module_name : str
        Name of the backend's module.
    """

    ARRAY_API_STRICT = "array_api_strict", "array_api_strict", "array_api_strict"
    NUMPY = "numpy", "numpy", "numpy"
    NUMPY_READONLY = "numpy_readonly", "numpy", "numpy"
    CUPY = "cupy", "cupy", "cupy"
    TORCH = "torch", "torch", "torch"
    DASK_ARRAY = "dask.array", "dask", "dask.array"
    SPARSE = "sparse", "pydata_sparse", "sparse"
    JAX_NUMPY = "jax.numpy", "jax", "jax.numpy"

    def __new__(
        cls, value: str, _library_name: str, _module_name: str
    ):  # numpydoc ignore=GL08
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(
        self,
        value: str,  # noqa: ARG002  # pylint: disable=unused-argument
        library_name: str,
        module_name: str,
    ):  # numpydoc ignore=GL08
        self.library_name = library_name
        self.module_name = module_name

    def __str__(self) -> str:  # type: ignore[explicit-override]  # pyright: ignore[reportImplicitOverride]  # numpydoc ignore=RT01
        """Pretty-print parameterized test names."""
        return cast(str, self.value)

    def is_namespace(self, xp: ModuleType) -> bool:
        """
        Call the corresponding is_namespace function.

        Parameters
        ----------
        xp : array_namespace
            Array namespace to check.

        Returns
        -------
        bool
            ``True`` if xp matches the namespace, ``False`` otherwise.
        """
        is_namespace_func = getattr(_compat, f"is_{self.library_name}_namespace")
        is_namespace_func = cast(Callable[[ModuleType], bool], is_namespace_func)
        return is_namespace_func(xp)
