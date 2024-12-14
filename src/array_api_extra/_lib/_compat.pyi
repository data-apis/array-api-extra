"""Static type stubs for `_compat.py`."""

from types import ModuleType

from ._typing import Array, Device

# pylint: disable=missing-class-docstring,unused-argument

class ArrayModule(ModuleType):  # numpydoc ignore=GL08
    def device(self, x: Array, /) -> Device: ...  # numpydoc ignore=GL08

def array_namespace(
    *xs: Array,
    api_version: str | None = None,
    use_compat: bool | None = None,
) -> ArrayModule: ...  # numpydoc ignore=GL08
def device(x: Array, /) -> Device: ...  # numpydoc ignore=GL08
