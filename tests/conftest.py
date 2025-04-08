"""Pytest fixtures."""

from collections.abc import Callable, Generator
from contextlib import suppress
from functools import cache, partial, wraps
from types import ModuleType
from typing import ParamSpec, TypeVar, cast

import numpy as np
import pytest

from array_api_extra._lib import Backend
from array_api_extra._lib._testing import xfail
from array_api_extra._lib._utils._compat import array_namespace
from array_api_extra._lib._utils._compat import device as get_device
from array_api_extra._lib._utils._typing import Device
from array_api_extra.testing import patch_lazy_xp_functions

T = TypeVar("T")
P = ParamSpec("P")

NUMPY_VERSION = tuple(int(v) for v in np.__version__.split(".")[2])
np_compat = array_namespace(np.empty(0))  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


@pytest.fixture(params=tuple(Backend))
def library(request: pytest.FixtureRequest) -> Backend:  # numpydoc ignore=PR01,RT03
    """
    Parameterized fixture that iterates on all libraries.

    Returns
    -------
    The current Backend enum.
    """
    elem = cast(Backend, request.param)

    for marker_name, skip_or_xfail in (
        ("skip_xp_backend", pytest.skip),
        ("xfail_xp_backend", partial(xfail, request)),
    ):
        for marker in request.node.iter_markers(marker_name):
            library = marker.kwargs.get("library") or marker.args[0]  # type: ignore[no-untyped-usage]
            if not isinstance(library, Backend):
                msg = f"argument of {marker_name} must be a Backend enum"
                raise TypeError(msg)
            if library == elem:
                reason = str(library)
                with suppress(KeyError):
                    reason += ":" + cast(str, marker.kwargs["reason"])
                skip_or_xfail(reason=reason)

    return elem


class NumPyReadOnly:
    """
    Variant of array_api_compat.numpy producing read-only arrays.

    Read-only NumPy arrays fail on `__iadd__` etc., whereas read-only libraries such as
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


@cache
def _jax_cuda_device() -> Device | None:
    """Return a CUDA device for JAX, if available."""
    import jax

    try:
        return jax.devices("cuda")[0]
    except Exception:
        return None


@cache
def _torch_cuda_device() -> Device | None:
    """Return a CUDA device for PyTorch, if available."""
    import torch

    try:
        return torch.empty((0,), device=torch.device("cuda")).device
    except Exception:
        return None


@pytest.fixture
def xp(
    library: Backend, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Generator[ModuleType]:  # numpydoc ignore=PR01,RT03
    """
    Parameterized fixture that iterates on all libraries.

    Returns
    -------
    The current array namespace.
    """
    if library == Backend.NUMPY_READONLY:
        yield NumPyReadOnly()  # type: ignore[misc]  # pyright: ignore[reportReturnType]
        return

    if library.like(Backend.ARRAY_API_STRICT) and NUMPY_VERSION < (1, 26):
        pytest.skip("array_api_strict is untested on NumPy <1.26")

    xp = pytest.importorskip(library.modname)
    # Possibly wrap module with array_api_compat
    xp = array_namespace(xp.empty(0))

    if library == Backend.ARRAY_API_STRICTEST:
        with xp.ArrayAPIStrictFlags(
            boolean_indexing=False,
            data_dependent_shapes=False,
            # writeable=False,  # TODO implement in array-api-strict
            # lazy=True,  # TODO implement in array-api-strict
            enabled_extensions=(),
        ):
            yield xp
        return

    # On Dask and JAX, monkey-patch all functions tagged by `lazy_xp_function`
    # in the global scope of the module containing the test function.
    patch_lazy_xp_functions(request, monkeypatch, xp=xp)

    if library.like(Backend.JAX):
        import jax

        # suppress unused-ignore to run mypy in -e lint as well as -e dev
        jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call,unused-ignore]

        if library == Backend.JAX_GPU:
            device = _jax_cuda_device()
            if device is None:
                pytest.skip("no cuda device available")
        else:
            device = jax.devices("cpu")[0]
        jax.config.update("jax_default_device", device)

    elif library == Backend.TORCH_GPU:
        if _torch_cuda_device() is None:
            pytest.skip("no cuda device available")
        xp.set_default_device("cuda")

    elif library == Backend.TORCH:  # CPU
        xp.set_default_device("cpu")

    yield xp


@pytest.fixture(params=[Backend.DASK])  # Can select the test with `pytest -k dask`
def da(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> ModuleType:  # numpydoc ignore=PR01,RT01
    """Variant of the `xp` fixture that only yields dask.array."""
    xp = pytest.importorskip("dask.array")
    xp = array_namespace(xp.empty(0))
    patch_lazy_xp_functions(request, monkeypatch, xp=xp)
    return xp


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
