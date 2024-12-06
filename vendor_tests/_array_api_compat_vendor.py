# This file is a hook imported by src/array_api_extra/_lib/_compat.py
from __future__ import annotations

from .array_api_compat import *  # noqa: F403
from .array_api_compat import array_namespace as array_namespace_compat


def array_namespace(*xs, **kwargs):
    return array_namespace_compat(*xs, **kwargs)
