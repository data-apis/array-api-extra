"""Configure doctests."""

import warnings
from collections.abc import Iterator
from contextlib import contextmanager

from scipy_doctest.conftest import dt_config


@contextmanager
def _doctest_context(  # numpydoc ignore=PR01
    _test: object | None = None,
) -> Iterator[None]:
    """
    Suppress expected warnings in public API doctests.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`xpx\.(broadcast_shapes|expand_dims)` is deprecated.*",
            category=DeprecationWarning,
        )
        yield


# Use this instead of `--doctest-only-doctests=true`, which ignores every path
# containing "tests", including installs in the `tests` pixi environment.
dt_config.pytest_extra_ignore.append("array_api_extra/tests")
dt_config.rtol = 1e-7
dt_config.strict_check = True
dt_config.user_context_mgr = _doctest_context
