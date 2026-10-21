"""Helpers from array-api-compat."""
# Allow packages that vendor both `array-api-extra` and
# `array-api-compat` to override the import location

from collections.abc import Callable
from functools import cache
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar, Protocol, cast

if TYPE_CHECKING:  # pragma: no cover
    from typing_extensions import override
else:

    def override(func):
        return func


# pylint: disable=duplicate-code,redefined-outer-name
try:
    from ..._array_api_compat_vendor import (
        array_namespace as _array_namespace,
    )
    from ..._array_api_compat_vendor import (
        device as _device,
    )
    from ..._array_api_compat_vendor import (
        is_array_api_obj,
        is_array_api_strict_namespace,
        is_cupy_array,
        is_cupy_namespace,
        is_dask_array,
        is_dask_namespace,
        is_jax_array,
        is_jax_namespace,
        is_lazy_array,
        is_numpy_array,
        is_numpy_namespace,
        is_pydata_sparse_array,
        is_pydata_sparse_namespace,
        is_torch_array,
        is_torch_namespace,
        is_writeable_array,
        size,
    )
    from ..._array_api_compat_vendor import (
        to_device as _to_device,
    )
except ImportError:
    from array_api_compat import (
        array_namespace as _array_namespace,
    )
    from array_api_compat import (
        device as _device,
    )
    from array_api_compat import (
        is_array_api_obj,
        is_array_api_strict_namespace,
        is_cupy_array,
        is_cupy_namespace,
        is_dask_array,
        is_dask_namespace,
        is_jax_array,
        is_jax_namespace,
        is_lazy_array,
        is_numpy_array,
        is_numpy_namespace,
        is_pydata_sparse_array,
        is_pydata_sparse_namespace,
        is_torch_array,
        is_torch_namespace,
        is_writeable_array,
        size,
    )
    from array_api_compat import (
        to_device as _to_device,
    )


class _NamespaceInfo(Protocol):  # numpydoc ignore=GL08
    def default_dtypes(self) -> object: ...  # numpydoc ignore=GL08


class _MLXNamespaceInfo:  # numpydoc ignore=GL08
    _info: _NamespaceInfo

    def __init__(self, info: _NamespaceInfo) -> None:  # numpydoc ignore=GL08
        self._info = info

    def __getattr__(self, name: str) -> object:  # numpydoc ignore=GL08
        return getattr(self._info, name)

    def default_dtypes(
        self, *, device: object = None
    ) -> object:  # numpydoc ignore=GL08
        _ = device
        return self._info.default_dtypes()


class _MLXNamespace(ModuleType):  # numpydoc ignore=GL08
    _device_functions: ClassVar[set[str]] = {
        "arange",
        "asarray",
        "empty",
        "eye",
        "full",
        "linspace",
        "ones",
        "zeros",
    }

    _namespace: ModuleType

    def __init__(self, namespace: ModuleType) -> None:  # numpydoc ignore=GL08
        super().__init__(namespace.__name__)
        self._namespace = namespace

    @override
    def __getattr__(self, name: str) -> object:  # numpydoc ignore=GL08
        if name == "__array_api_version__":
            return "2024.12"
        if name == "bool":
            return self._namespace.bool_
        if name == "__array_namespace_info__":
            get_info = cast(
                Callable[[], _NamespaceInfo], getattr(self._namespace, name)
            )
            return lambda: _MLXNamespaceInfo(get_info())
        if name == "signbit":
            return lambda x: x < 0
        function = getattr(self._namespace, name)
        if name not in {"argsort", "astype", "result_type", "sort"} and (
            name not in self._device_functions
        ):
            return function

        def compatible_call(  # numpydoc ignore=GL08
            *args: object,
            device: object = None,
            copy: bool | None = None,
            **kwargs: object,
        ) -> object:
            if name == "full" and "fill_value" in kwargs:
                args = (*args, kwargs.pop("fill_value"))
            if name == "asarray" and copy is not None:
                kwargs["copy"] = copy
            if name == "asarray" and "dtype" not in kwargs:
                input_array = args[0] if args else kwargs.get("a")
                input_dtype = getattr(input_array, "dtype", None)
                dtype_name = getattr(input_dtype, "name", None)
                if dtype_name is not None and hasattr(self._namespace, dtype_name):
                    kwargs["dtype"] = getattr(self._namespace, dtype_name)
            if name == "astype":
                _ = copy
                return function(*args, **kwargs)
            if name in {"argsort", "sort"}:
                _ = kwargs.pop("stable", None)
            if name == "result_type":
                import mlx.core as mx

                converted_args: list[object] = []
                for value in args:
                    if isinstance(value, int | float | complex | bool):
                        converted_args.append(cast(object, mx.asarray(value)))
                    else:
                        converted_args.append(value)
                args = tuple(converted_args)
            if device is None:
                return function(*args, **kwargs)
            import mlx.core as mx

            with mx.stream(device):
                return function(*args, **kwargs)

        return compatible_call


@cache
def _wrap_mlx_namespace(namespace: ModuleType) -> ModuleType:  # numpydoc ignore=GL08
    return _MLXNamespace(namespace)


def array_namespace(  # numpydoc ignore=GL08
    *xs: object, api_version: str | None = None, use_compat: bool | None = None
) -> ModuleType:
    namespace = cast(
        ModuleType,
        _array_namespace(*xs, api_version=api_version, use_compat=use_compat),
    )
    if namespace.__name__ == "mlx.core":
        return _wrap_mlx_namespace(namespace)
    return namespace


def device(x: object, /) -> object:  # numpydoc ignore=GL08
    if type(x).__module__.startswith("mlx."):
        import mlx.core as mx

        return mx.default_device()
    return _device(x)


def to_device(  # numpydoc ignore=GL08
    x: object, device: object, /, *, stream: object = None
) -> object:
    if type(x).__module__.startswith("mlx."):
        import mlx.core as mx

        if device == "cpu":
            device = mx.cpu
        elif device == "gpu":
            device = mx.gpu
        with mx.stream(device):
            return mx.array(x)
    return _to_device(x, device, stream=stream)


__all__ = [
    "array_namespace",
    "device",
    "is_array_api_obj",
    "is_array_api_strict_namespace",
    "is_cupy_array",
    "is_cupy_namespace",
    "is_dask_array",
    "is_dask_namespace",
    "is_jax_array",
    "is_jax_namespace",
    "is_lazy_array",
    "is_numpy_array",
    "is_numpy_namespace",
    "is_pydata_sparse_array",
    "is_pydata_sparse_namespace",
    "is_torch_array",
    "is_torch_namespace",
    "is_writeable_array",
    "size",
    "to_device",
]
