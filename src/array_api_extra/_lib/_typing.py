from __future__ import annotations  # https://github.com/pylint-dev/pylint/pull/9990

import typing
from collections.abc import Mapping
from types import ModuleType
from typing import Any, Protocol

if typing.TYPE_CHECKING:
    from typing_extensions import override

    # To be changed to a Protocol later (see data-apis/array-api#589)
    Untyped = Any  # type: ignore[no-any-explicit]
    Array = Untyped
    Device = Untyped
    Index = Untyped

    class CanAt(Protocol):
        @property
        def at(self) -> Mapping[Index, Untyped]: ...

else:

    def no_op_decorator(f):  # pyright: ignore[reportUnreachable]
        return f

    override = no_op_decorator

    CanAt = object

__all__ = ["ModuleType", "override"]
if typing.TYPE_CHECKING:
    __all__ += ["Array", "Device", "Index", "Untyped"]
