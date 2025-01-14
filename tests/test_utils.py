import numpy as np
import pytest

from array_api_extra._lib._compat import device as get_device
from array_api_extra._lib._testing import xp_assert_equal
from array_api_extra._lib._typing import Array, Device, ModuleType
from array_api_extra._lib._utils import in1d

from .conftest import Library

# mypy: disable-error-code=no-untyped-usage


class TestIn1D:
    @pytest.mark.skip_xp_backend(Library.DASK_ARRAY, reason="no argsort")
    @pytest.mark.skip_xp_backend(Library.SPARSE, reason="no unique_inverse, no device")
    # cover both code paths
    @pytest.mark.parametrize("x2", [np.arange(9), np.arange(15)])
    def test_no_invert_assume_unique(self, xp: ModuleType, x2: Array):
        x1 = xp.asarray([3, 8, 20])
        x2 = xp.asarray(x2)
        expected = xp.asarray([True, True, False])
        actual = in1d(x1, x2)
        xp_assert_equal(actual, expected)

    @pytest.mark.skip_xp_backend(Library.SPARSE, reason="no device")
    def test_device(self, xp: ModuleType, device: Device):
        x1 = xp.asarray([3, 8, 20], device=device)
        x2 = xp.asarray([2, 3, 4], device=device)
        assert get_device(in1d(x1, x2)) == device

    @pytest.mark.skip_xp_backend(Library.SPARSE, reason="no arange, no device")
    def test_xp(self, xp: ModuleType):
        x1 = xp.asarray([1, 6])
        x2 = xp.arange(5)
        expected = xp.asarray([True, False])
        actual = in1d(x1, x2, xp=xp)
        xp_assert_equal(actual, expected)
