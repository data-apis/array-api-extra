"""Pytest fixtures."""

from collections.abc import Callable
from functools import wraps
from types import ModuleType
from typing import ParamSpec, TypeVar, cast

import numpy as np
import pytest

from array_api_extra._lib import Backend
from array_api_extra._lib._utils._compat import array_namespace
from array_api_extra._lib._utils._compat import device as get_device
from array_api_extra._lib._utils._typing import Device

T = TypeVar("T")
P = ParamSpec("P")

np_compat = array_namespace(np.empty(0))


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


class NumPyReadOnly:
    """
    Variant of array_api_compat.numpy producing read-only arrays.

    Read-only numpy arrays fail on `__iadd__` etc., whereas read-only libraries such as
    JAX and Sparse simply don't define those methods, which makes calls to `+=` fall
    back to `__add__`.

    Note that this is not a full read-only Array API library. Notably,
    `array_namespace(x)` returns array_api_compat.numpy. This is actually the desired
    behaviour, so that when a tested function internally calls `xp =
    array_namespace(*args) or xp`, it will internally create writeable arrays.
    For this reason, tests that explicitly pass xp=xp to the tested functions may
    misbehave and should be skipped for NUMPY_READONLY.
    """

    def __getattr__(self, name: str) -> object:  # numpydoc ignore=PR01,RT01
        """Wrap all functions that return arrays to make their output read-only."""
        func = getattr(np_compat, name)
        if not callable(func) or isinstance(func, type):
            return func
        return self._wrap(func)

    @staticmethod
    def _wrap(func: Callable[P, T]) -> Callable[P, T]:  # numpydoc ignore=PR01,RT01
        """Wrap func to make all np.ndarrays it returns read-only."""

        def as_readonly(o: T) -> T:  # numpydoc ignore=PR01,RT01
            """Unset the writeable flag in o."""
            try:
                # Don't use is_numpy_array(o), as it includes np.generic
                if isinstance(o, np.ndarray):
                    o.flags.writeable = False
            except TypeError:
                # Cannot interpret as a data type
                return o

            # This works with namedtuples too
            if isinstance(o, tuple | list):
                return type(o)(*(as_readonly(i) for i in o))  # type: ignore[arg-type,return-value] # pyright: ignore[reportArgumentType,reportUnknownArgumentType]

            return o

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:  # numpydoc ignore=GL08
            return as_readonly(func(*args, **kwargs))

        return wrapper


@pytest.fixture
def xp(library: Backend) -> ModuleType:  # numpydoc ignore=PR01,RT03
    """
    Parameterized fixture that iterates on all libraries.

    Returns
    -------
    The current array namespace.
    """
    if library == Backend.NUMPY_READONLY:
        return NumPyReadOnly()  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
    xp = pytest.importorskip(library.value)
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
