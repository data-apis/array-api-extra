from __future__ import annotations  # https://github.com/pylint-dev/pylint/pull/9990

# data-apis/array-api-strict#6
import array_api_strict as xp  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
from numpy.testing import assert_array_equal

from array_api_extra._lib._utils import in1d


# some test coverage already provided by TestSetDiff1D
class TestIn1D:
    def test_no_invert_assume_unique(self):
        x1 = xp.asarray([1, 2, 3])
        x2 = xp.asarray([3, 4, 5])
        expected = xp.asarray([False, False, True])
        actual = in1d(x1, x2, xp=xp)
        assert_array_equal(actual, expected)
