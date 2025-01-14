"""Pytest fixtures."""

from enum import Enum
from typing import cast

import pytest

from array_api_extra._lib._compat import array_namespace
from array_api_extra._lib._compat import device as get_device
from array_api_extra._lib._typing import Device, ModuleType


class Library(Enum):
    """All array libraries explicitly tested by array-api-extra."""

    ARRAY_API_STRICT = "array_api_strict"
    NUMPY = "numpy"
    NUMPY_READONLY = "numpy_readonly"
    CUPY = "cupy"
    TORCH = "torch"
    DASK_ARRAY = "dask.array"
    SPARSE = "sparse"
    JAX_NUMPY = "jax.numpy"

    def __str__(self) -> str:  # type: ignore[explicit-override]  # pyright: ignore[reportImplicitOverride]  # numpydoc ignore=RT01
        """Pretty-print parameterized test names."""
        return self.value


@pytest.fixture(params=tuple(Library))
def library(request: pytest.FixtureRequest) -> Library:  # numpydoc ignore=PR01,RT03
    """
    Parameterized fixture that iterates on all libraries.

    Returns
    -------
    The current Library enum.
    """
    elem = cast(Library, request.param)

    for marker in request.node.iter_markers("skip_xp_backend"):
        skip_library = marker.kwargs.get("library") or marker.args[0]  # type: ignore[no-untyped-usage]
        if not isinstance(skip_library, Library):
            msg = "argument of skip_xp_backend must be a Library enum"
            raise TypeError(msg)
        if skip_library == elem:
            reason = cast(str, marker.kwargs.get("reason", "skip_xp_backend"))
            pytest.skip(reason=reason)

    return elem


@pytest.fixture
def xp(library: Library) -> ModuleType:  # numpydoc ignore=PR01,RT03
    """
    Parameterized fixture that iterates on all libraries.

    Returns
    -------
    The current array namespace.
    """
    name = "numpy" if library == Library.NUMPY_READONLY else library.value
    xp = pytest.importorskip(name)
    if library == Library.JAX_NUMPY:
        import jax  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

        jax.config.update("jax_enable_x64", True)

    # Possibly wrap module with array_api_compat
    return array_namespace(xp.empty(0))


@pytest.fixture
def device(
    library: Library, xp: ModuleType
) -> Device:  # numpydoc ignore=PR01,RT01,RT03
    """
    Return a valid device for the backend.

    Where possible, return a device that is not the default one.
    """
    if library == Library.ARRAY_API_STRICT:
        d = xp.Device("device1")
        assert get_device(xp.empty(0)) != d
        return d
    return get_device(xp.empty(0))
