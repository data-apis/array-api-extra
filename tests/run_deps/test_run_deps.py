"""Tests to run with only runtime dependencies."""

import importlib

import pytest

import array_api_extra.testing as xpt


def test_no_numpy() -> None:
    """Check `xpx.testing` assertion error message when NumPy is unavailable."""
    if importlib.util.find_spec("numpy") is not None:  # pyright: ignore[reportAttributeAccessIssue]
        pytest.skip("Test for when `numpy` is not importable.")
    with pytest.raises(ImportError, match=r"assertion.*require.*numpy"):
        xpt.assert_equal(1, 1)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
