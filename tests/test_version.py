from __future__ import annotations  # https://github.com/pylint-dev/pylint/pull/9990

import importlib.metadata

import array_api_extra as xpx


def test_version():
    assert importlib.metadata.version("array_api_extra") == xpx.__version__
