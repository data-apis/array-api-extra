import array_api_strict as xp
from numpy.testing import assert_array_equal


def test_vendor_compat():
    from ._array_api_compat_vendor import (  # type: ignore[attr-defined]
        array_namespace,
        device,
        is_jax_array,
        is_writeable_array,
    )

    x = xp.asarray([1, 2, 3])
    assert array_namespace(x) is xp
    device(x)
    assert not is_jax_array(x)
    assert is_writeable_array(x)


def test_vendor_extra():
    from .array_api_extra import atleast_nd

    x = xp.asarray(1)
    y = atleast_nd(x, ndim=0)
    assert_array_equal(y, x)


def test_vendor_extra_uses_vendor_compat():
    from ._array_api_compat_vendor import array_namespace as n1
    from .array_api_extra._lib._compat import array_namespace as n2

    assert n1 is n2
