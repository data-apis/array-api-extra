from types import ModuleType

import pytest

from array_api_extra._lib import Backend
from array_api_extra._lib._testing import xp_assert_equal
from array_api_extra._lib._utils._compat import device as get_device
from array_api_extra._lib._utils._helpers import in1d
from array_api_extra._lib._utils._typing import Device

# mypy: disable-error-code=no-untyped-usage


class TestIn1D:
    @pytest.mark.skip_xp_backend(Backend.DASK, reason="dask:no argsort")
    @pytest.mark.skip_xp_backend(
        Backend.SPARSE, reason="sparse:no unique_inverse, no device kwarg in asarray"
    )
    # cover both code paths
    @pytest.mark.parametrize("n", [9, 15])
    def test_no_invert_assume_unique(self, xp: ModuleType, n: int):
        x1 = xp.asarray([3, 8, 20])
        x2 = xp.arange(n)
        expected = xp.asarray([True, True, False])
        actual = in1d(x1, x2)
        xp_assert_equal(actual, expected)

    @pytest.mark.skip_xp_backend(
        Backend.SPARSE, reason="sparse: no device kwarg in asarray"
    )
    def test_device(self, xp: ModuleType, device: Device):
        x1 = xp.asarray([3, 8, 20], device=device)
        x2 = xp.asarray([2, 3, 4], device=device)
        assert get_device(in1d(x1, x2)) == device

    @pytest.mark.skip_xp_backend(
        Backend.NUMPY_READONLY, reason="numpy_readonly:explicit xp"
    )
    @pytest.mark.skip_xp_backend(
        Backend.SPARSE, reason="sparse:no arange, no device kwarg in asarray"
    )
    def test_xp(self, xp: ModuleType):
        x1 = xp.asarray([1, 6])
        x2 = xp.arange(5)
        expected = xp.asarray([True, False])
        actual = in1d(x1, x2, xp=xp)
        xp_assert_equal(actual, expected)
