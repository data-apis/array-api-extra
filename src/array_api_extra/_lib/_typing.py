"""Static typing helpers."""
# numpydoc ignore=GL08
# pylint: disable=duplicate-code

from types import ModuleType

Array = object
ArrayNamespace = ModuleType
DType = object
Device = object
GetIndex = object
SetIndex = object

__all__ = ["Array", "ArrayNamespace", "DType", "Device", "GetIndex", "SetIndex"]
