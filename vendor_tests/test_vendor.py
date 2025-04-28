import array_api_strict as xp
from numpy.testing import assert_array_equal


def test_vendor_compat():
    from ._array_api_compat_vendor import (  # type: ignore[attr-defined]
        array_namespace,
        device,
        is_array_api_obj,
        is_array_api_strict_namespace,
        is_cupy_array,
        is_cupy_namespace,
        is_dask_array,
        is_dask_namespace,
        is_jax_array,
        is_jax_namespace,
        is_lazy_array,
        is_numpy_array,
        is_numpy_namespace,
        is_pydata_sparse_array,
        is_pydata_sparse_namespace,
        is_torch_array,
        is_torch_namespace,
        is_writeable_array,
        size,
        to_device,
    )

    x = xp.asarray([1, 2, 3])
    assert array_namespace(x) is xp
    to_device(x, device(x))
    assert is_array_api_obj(x)
    assert is_array_api_strict_namespace(xp)
    assert not is_cupy_array(x)
    assert not is_cupy_namespace(xp)
    assert not is_dask_array(x)
    assert not is_dask_namespace(xp)
    assert not is_jax_array(x)
    assert not is_jax_namespace(xp)
    assert not is_lazy_array(x)
    assert not is_numpy_array(x)
    assert not is_numpy_namespace(xp)
    assert not is_pydata_sparse_array(x)
    assert not is_pydata_sparse_namespace(xp)
    assert not is_torch_array(x)
    assert not is_torch_namespace(xp)
    assert is_writeable_array(x)
    assert size(x) == 3


def test_vendor_extra():
    from .array_api_extra import atleast_nd

    x = xp.asarray(1)
    y = atleast_nd(x, ndim=0)
    assert_array_equal(y, x)


def test_vendor_extra_testing():
    from .array_api_extra.testing import lazy_xp_function

    def f(x):
        return x

    lazy_xp_function(f)


def test_vendor_extra_uses_vendor_compat():
    from ._array_api_compat_vendor import array_namespace as n1
    from .array_api_extra._lib._utils._compat import array_namespace as n2

    assert n1 is n2
