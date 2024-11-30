from __future__ import annotations  # https://github.com/pylint-dev/pylint/pull/9990

import typing

# data-apis/array-api-strict#6
import array_api_strict as xp  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
import pytest
from numpy.testing import assert_array_equal

from array_api_extra._lib._utils import in1d

if typing.TYPE_CHECKING:
    from array_api_extra._lib._typing import Array


# some test coverage already provided by TestSetDiff1D
class TestIn1D:
    # cover both code paths
    @pytest.mark.parametrize("x2", [xp.arange(9), xp.arange(15)])
    def test_no_invert_assume_unique(self, x2: Array):
        x1 = xp.asarray([3, 8, 20])
        expected = xp.asarray([True, True, False])
        actual = in1d(x1, x2, xp=xp)
        assert_array_equal(actual, expected)
