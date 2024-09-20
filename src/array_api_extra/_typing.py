from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    Array = Any  # To be changed to a Protocol later (see array-api#589)

__all__ = ["Array", "ModuleType"]
