"""Pytest fixtures."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from functools import wraps
from typing import ParamSpec, TypeVar, cast

import numpy as np
import pytest

from array_api_extra._lib._compat import array_namespace
from array_api_extra._lib._compat import device as get_device
from array_api_extra._lib._typing import Device, ModuleType

T = TypeVar("T")
P = ParamSpec("P")

np_compat = array_namespace(np.empty(0))


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


class NumPyReadOnly:
    """
    Variant of array_api_compat.numpy producing read-only arrays.

    Note that this is not a full read-only Array API library. Notably,
    array_namespace(x) returns array_api_compat.numpy, and as a consequence array
    creation functions invoked internally by the tested functions will return
    writeable arrays, as long as you don't explicitly pass xp=xp.
    For this reason, tests that do pass xp=xp may misbehave and should be skipped
    for NUMPY_READONLY.
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

        def as_readonly(o: T, seen: set[int]) -> T:  # numpydoc ignore=PR01,RT01
            """Unset the writeable flag in o."""
            if id(o) in seen:
                return o
            seen.add(id(o))

            try:
                # Don't use is_numpy_array(o), as it includes np.generic
                if isinstance(o, np.ndarray):
                    o.flags.writeable = False
            except TypeError:
                # Cannot interpret as a data type
                return o

            # This works with namedtuples too
            if isinstance(o, tuple | list):
                return type(o)(*(as_readonly(i, seen) for i in o))  # type: ignore[arg-type,return-value] # pyright: ignore[reportArgumentType,reportUnknownArgumentType]

            return o

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:  # numpydoc ignore=GL08
            return as_readonly(func(*args, **kwargs), seen=set())

        return wrapper


@pytest.fixture
def xp(library: Library) -> ModuleType:  # numpydoc ignore=PR01,RT03
    """
    Parameterized fixture that iterates on all libraries.

    Returns
    -------
    The current array namespace.
    """
    if library == Library.NUMPY_READONLY:
        return NumPyReadOnly()  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
    xp = pytest.importorskip(library.value)
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
