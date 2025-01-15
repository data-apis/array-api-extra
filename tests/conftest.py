"""Pytest fixtures."""

from types import ModuleType
from typing import cast

import pytest

from array_api_extra._lib import Backend
from array_api_extra._lib._utils._compat import array_namespace
from array_api_extra._lib._utils._compat import device as get_device
from array_api_extra._lib._utils._typing import Device


@pytest.fixture(params=tuple(Backend))
def library(request: pytest.FixtureRequest) -> Backend:  # numpydoc ignore=PR01,RT03
    """
    Parameterized fixture that iterates on all libraries.

    Returns
    -------
    The current Backend enum.
    """
    elem = cast(Backend, request.param)

    for marker in request.node.iter_markers("skip_xp_backend"):
        skip_library = marker.kwargs.get("library") or marker.args[0]  # type: ignore[no-untyped-usage]
        if not isinstance(skip_library, Backend):
            msg = "argument of skip_xp_backend must be a Backend enum"
            raise TypeError(msg)
        if skip_library == elem:
            reason = cast(str, marker.kwargs.get("reason", "skip_xp_backend"))
            pytest.skip(reason=reason)

    return elem


@pytest.fixture
def xp(library: Backend) -> ModuleType:  # numpydoc ignore=PR01,RT03
    """
    Parameterized fixture that iterates on all libraries.

    Returns
    -------
    The current array namespace.
    """
    xp = pytest.importorskip(library.module_name)
    if library == Backend.JAX_NUMPY:
        import jax  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

        jax.config.update("jax_enable_x64", True)

    # Possibly wrap module with array_api_compat
    return array_namespace(xp.empty(0))


@pytest.fixture
def device(
    library: Backend, xp: ModuleType
) -> Device:  # numpydoc ignore=PR01,RT01,RT03
    """
    Return a valid device for the backend.

    Where possible, return a device that is not the default one.
    """
    if library == Backend.ARRAY_API_STRICT:
        d = xp.Device("device1")
        assert get_device(xp.empty(0)) != d
        return d
    return get_device(xp.empty(0))
