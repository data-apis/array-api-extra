from types import ModuleType

import numpy as np
import pytest

from array_api_extra._lib import Backend
from array_api_extra._lib._testing import xp_assert_equal
from array_api_extra._lib._utils._compat import device as get_device
from array_api_extra._lib._utils._helpers import asarrays, in1d
from array_api_extra._lib._utils._typing import Device
from array_api_extra.testing import lazy_xp_function

# mypy: disable-error-code=no-untyped-usage

# FIXME calls xp.unique_values without size
lazy_xp_function(in1d, jax_jit=False, static_argnames=("assume_unique", "invert", "xp"))


class TestIn1D:
    @pytest.mark.xfail_xp_backend(
        Backend.SPARSE, reason="no unique_inverse, no device kwarg in asarray()"
    )
    # cover both code paths
    @pytest.mark.parametrize(
        "n",
        [
            pytest.param(9, id="fast path"),
            pytest.param(
                15,
                id="slow path",
                marks=pytest.mark.xfail_xp_backend(
                    Backend.DASK, reason="NaN-shaped array"
                ),
            ),
        ],
    )
    def test_no_invert_assume_unique(self, xp: ModuleType, n: int):
        x1 = xp.asarray([3, 8, 20])
        x2 = xp.arange(n)
        expected = xp.asarray([True, True, False])
        actual = in1d(x1, x2)
        xp_assert_equal(actual, expected)

    @pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no device kwarg in asarray")
    def test_device(self, xp: ModuleType, device: Device):
        x1 = xp.asarray([3, 8, 20], device=device)
        x2 = xp.asarray([2, 3, 4], device=device)
        assert get_device(in1d(x1, x2)) == device

    @pytest.mark.skip_xp_backend(Backend.NUMPY_READONLY, reason="xp=xp")
    @pytest.mark.xfail_xp_backend(
        Backend.SPARSE, reason="no arange, no device kwarg in asarray"
    )
    def test_xp(self, xp: ModuleType):
        x1 = xp.asarray([1, 6])
        x2 = xp.arange(5)
        expected = xp.asarray([True, False])
        actual = in1d(x1, x2, xp=xp)
        xp_assert_equal(actual, expected)


@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no isdtype")
@pytest.mark.parametrize(
    ("dtype", "b", "defined"),
    [
        # Well-defined cases of dtype promotion from Python scalar to Array
        # bool vs. bool
        ("bool", True, True),
        # int vs. xp.*int*, xp.float*, xp.complex*
        ("int16", 1, True),
        ("uint8", 1, True),
        ("float32", 1, True),
        ("float64", 1, True),
        ("complex64", 1, True),
        ("complex128", 1, True),
        # float vs. xp.float, xp.complex
        ("float32", 1.0, True),
        ("float64", 1.0, True),
        ("complex64", 1.0, True),
        ("complex128", 1.0, True),
        # complex vs. xp.complex
        ("complex64", 1.0j, True),
        ("complex128", 1.0j, True),
        # Undefined cases
        ("bool", 1, False),
        ("int64", 1.0, False),
        ("float64", 1.0j, False),
    ],
)
def test_asarrays_array_vs_scalar(
    dtype: str, b: int | float | complex, defined: bool, xp: ModuleType
):
    a = xp.asarray(1, dtype=getattr(xp, dtype))

    xa, xb = asarrays(a, b, xp)
    assert xa.dtype == a.dtype
    if defined:
        assert xb.dtype == a.dtype
    else:
        assert xb.dtype == xp.asarray(b).dtype

    xbr, xar = asarrays(b, a, xp)
    assert xar.dtype == xa.dtype
    assert xbr.dtype == xb.dtype


def test_asarrays_scalar_vs_scalar(xp: ModuleType):
    a, b = asarrays(1, 2.2, xp=xp)
    assert a.dtype == xp.asarray(1).dtype  # Default dtype
    assert b.dtype == xp.asarray(2.2).dtype  # Default dtype; not broadcasted


ALL_TYPES = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "bool",
)


@pytest.mark.parametrize("a_type", ALL_TYPES)
@pytest.mark.parametrize("b_type", ALL_TYPES)
def test_asarrays_array_vs_array(a_type: str, b_type: str, xp: ModuleType):
    """
    Test that when both inputs of asarray are already Array API objects,
    they are returned unchanged.
    """
    a = xp.asarray(1, dtype=getattr(xp, a_type))
    b = xp.asarray(1, dtype=getattr(xp, b_type))
    xa, xb = asarrays(a, b, xp)
    assert xa.dtype == a.dtype
    assert xb.dtype == b.dtype


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_asarrays_numpy_generics(dtype: type):
    """
    Test special case of np.float64 and np.complex128,
    which are subclasses of float and complex.
    """
    a = dtype(0)
    xa, xb = asarrays(a, 0, xp=np)
    assert xa.dtype == dtype
    assert xb.dtype == dtype
