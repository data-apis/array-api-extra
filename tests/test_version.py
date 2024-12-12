import importlib.metadata

import array_api_extra as xpx


def test_version():
    assert importlib.metadata.version("array_api_extra") == xpx.__version__
