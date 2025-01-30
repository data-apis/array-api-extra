from types import ModuleType

import numpy as np
import pytest

from array_api_extra._lib import Backend
from array_api_extra._lib._utils._helpers import asarrays

# mypy: disable-error-code=no-untyped-usage


@pytest.mark.skip_xp_backend(Backend.SPARSE, reason="no isdtype")
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
