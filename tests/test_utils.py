# data-apis/array-api-strict#6
import array_api_strict as xp  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
import pytest
from numpy.testing import assert_array_equal

from array_api_extra._lib._typing import Array
from array_api_extra._lib._utils import in1d


# some test coverage already provided by TestSetDiff1D
class TestIn1D:
    # cover both code paths
    @pytest.mark.parametrize("x2", [xp.arange(9), xp.arange(15)])
    def test_no_invert_assume_unique(self, x2: Array):
        x1 = xp.asarray([3, 8, 20])
        expected = xp.asarray([True, True, False])
        actual = in1d(x1, x2)
        assert_array_equal(actual, expected)

    def test_device(self):
        device = xp.Device("device1")
        x1 = xp.asarray([3, 8, 20], device=device)
        x2 = xp.asarray([2, 3, 4], device=device)
        assert in1d(x1, x2).device == device

    def test_xp(self):
        x1 = xp.asarray([1, 6])
        x2 = xp.arange(5)
        expected = xp.asarray([True, False])
        actual = in1d(x1, x2, xp=xp)
        assert_array_equal(actual, expected)
