import pytest

from array_api_extra import default_dtype
from array_api_extra._lib._typing import ArrayNamespace, Device
from array_api_extra.testing import lazy_xp_function

lazy_xp_function(default_dtype)


class TestDefaultDType:
    def test_basic(self, xp: ArrayNamespace):
        assert default_dtype(xp) == xp.empty(0).dtype

    def test_kind(self, xp: ArrayNamespace):
        assert default_dtype(xp, "real floating") == xp.empty(0).dtype
        assert default_dtype(xp, "complex floating") == (xp.empty(0) * 1j).dtype
        assert default_dtype(xp, "integral") == xp.int64
        assert default_dtype(xp, "indexing") == xp.int64

        with pytest.raises(ValueError, match="Unknown kind"):
            _ = default_dtype(xp, "foo")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

    def test_device(self, xp: ArrayNamespace, device: Device):
        # Note: at the moment there are no known namespaces with
        # device-specific default dtypes.
        assert default_dtype(xp, device=None) == xp.empty(0).dtype
        assert default_dtype(xp, device=device) == xp.empty(0).dtype

    def test_torch(self, torch: ArrayNamespace):
        xp = torch
        xp.set_default_dtype(xp.float64)
        assert default_dtype(xp) == xp.float64
        assert default_dtype(xp, "real floating") == xp.float64
        assert default_dtype(xp, "complex floating") == xp.complex128

        xp.set_default_dtype(xp.float32)
        assert default_dtype(xp) == xp.float32
        assert default_dtype(xp, "real floating") == xp.float32
        assert default_dtype(xp, "complex floating") == xp.complex64
