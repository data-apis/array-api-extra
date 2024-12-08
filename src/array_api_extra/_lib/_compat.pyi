from types import ModuleType

from ._typing import Array, Device

class ArrayModule(ModuleType):
    def device(self, x: Array, /) -> Device: ...

def array_namespace(
    *xs: Array,
    api_version: str | None = None,
    use_compat: bool | None = None,
) -> ArrayModule: ...
def device(x: Array, /) -> Device: ...
