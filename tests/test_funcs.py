import contextlib
import warnings
from types import ModuleType

import numpy as np
import pytest

from array_api_extra import (
    at,
    atleast_nd,
    cov,
    create_diagonal,
    expand_dims,
    isclose,
    kron,
    nunique,
    pad,
    setdiff1d,
    sinc,
)
from array_api_extra._lib import Backend
from array_api_extra._lib._testing import xp_assert_close, xp_assert_equal
from array_api_extra._lib._utils._compat import device as get_device
from array_api_extra._lib._utils._typing import Array, Device
from array_api_extra.testing import lazy_xp_function

# some xp backends are untyped
# mypy: disable-error-code=no-untyped-def

lazy_xp_function(atleast_nd, static_argnames=("ndim", "xp"))
lazy_xp_function(cov, static_argnames="xp")
# FIXME .device attribute https://github.com/data-apis/array-api-compat/pull/238
lazy_xp_function(create_diagonal, jax_jit=False, static_argnames=("offset", "xp"))
lazy_xp_function(expand_dims, static_argnames=("axis", "xp"))
lazy_xp_function(kron, static_argnames="xp")
lazy_xp_function(nunique, static_argnames="xp")
lazy_xp_function(pad, static_argnames=("pad_width", "mode", "constant_values", "xp"))
# FIXME calls in1d which calls xp.unique_values without size
lazy_xp_function(setdiff1d, jax_jit=False, static_argnames=("assume_unique", "xp"))
# FIXME .device attribute https://github.com/data-apis/array-api-compat/pull/238
lazy_xp_function(sinc, jax_jit=False, static_argnames="xp")


@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no expand_dims")
class TestAtLeastND:
    def test_0D(self, xp: ModuleType):
        x = xp.asarray(1.0)

        y = atleast_nd(x, ndim=0)
        xp_assert_equal(y, x)

        y = atleast_nd(x, ndim=1)
        xp_assert_equal(y, xp.ones((1,)))

        y = atleast_nd(x, ndim=5)
        xp_assert_equal(y, xp.ones((1, 1, 1, 1, 1)))

    def test_1D(self, xp: ModuleType):
        x = xp.asarray([0, 1])

        y = atleast_nd(x, ndim=0)
        xp_assert_equal(y, x)

        y = atleast_nd(x, ndim=1)
        xp_assert_equal(y, x)

        y = atleast_nd(x, ndim=2)
        xp_assert_equal(y, xp.asarray([[0, 1]]))

        y = atleast_nd(x, ndim=5)
        xp_assert_equal(y, xp.reshape(xp.arange(2), (1, 1, 1, 1, 2)))

    def test_2D(self, xp: ModuleType):
        x = xp.asarray([[3.0]])

        y = atleast_nd(x, ndim=0)
        xp_assert_equal(y, x)

        y = atleast_nd(x, ndim=2)
        xp_assert_equal(y, x)

        y = atleast_nd(x, ndim=3)
        xp_assert_equal(y, 3 * xp.ones((1, 1, 1)))

        y = atleast_nd(x, ndim=5)
        xp_assert_equal(y, 3 * xp.ones((1, 1, 1, 1, 1)))

    def test_5D(self, xp: ModuleType):
        x = xp.ones((1, 1, 1, 1, 1))

        y = atleast_nd(x, ndim=0)
        xp_assert_equal(y, x)

        y = atleast_nd(x, ndim=4)
        xp_assert_equal(y, x)

        y = atleast_nd(x, ndim=5)
        xp_assert_equal(y, x)

        y = atleast_nd(x, ndim=6)
        xp_assert_equal(y, xp.ones((1, 1, 1, 1, 1, 1)))

        y = atleast_nd(x, ndim=9)
        xp_assert_equal(y, xp.ones((1, 1, 1, 1, 1, 1, 1, 1, 1)))

    def test_device(self, xp: ModuleType, device: Device):
        x = xp.asarray([1, 2, 3], device=device)
        assert get_device(atleast_nd(x, ndim=2)) == device

    def test_xp(self, xp: ModuleType):
        x = xp.asarray(1.0)
        y = atleast_nd(x, ndim=1, xp=xp)
        xp_assert_equal(y, xp.ones((1,)))


@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no isdtype")
class TestCov:
    def test_basic(self, xp: ModuleType):
        xp_assert_close(
            cov(xp.asarray([[0, 2], [1, 1], [2, 0]]).T),
            xp.asarray([[1.0, -1.0], [-1.0, 1.0]], dtype=xp.float64),
        )

    def test_complex(self, xp: ModuleType):
        actual = cov(xp.asarray([[1, 2, 3], [1j, 2j, 3j]]))
        expect = xp.asarray([[1.0, -1.0j], [1.0j, 1.0]], dtype=xp.complex128)
        xp_assert_close(actual, expect)

    def test_empty(self, xp: ModuleType):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", RuntimeWarning)
            xp_assert_equal(cov(xp.asarray([])), xp.asarray(xp.nan, dtype=xp.float64))
            xp_assert_equal(
                cov(xp.reshape(xp.asarray([]), (0, 2))),
                xp.reshape(xp.asarray([], dtype=xp.float64), (0, 0)),
            )
            xp_assert_equal(
                cov(xp.reshape(xp.asarray([]), (2, 0))),
                xp.asarray([[xp.nan, xp.nan], [xp.nan, xp.nan]], dtype=xp.float64),
            )

    def test_combination(self, xp: ModuleType):
        x = xp.asarray([-2.1, -1, 4.3])
        y = xp.asarray([3, 1.1, 0.12])
        X = xp.stack((x, y), axis=0)
        desired = xp.asarray([[11.71, -4.286], [-4.286, 2.144133]], dtype=xp.float64)
        xp_assert_close(cov(X), desired, rtol=1e-6)
        xp_assert_close(cov(x), xp.asarray(11.71, dtype=xp.float64))
        xp_assert_close(cov(y), xp.asarray(2.144133, dtype=xp.float64), rtol=1e-6)

    def test_device(self, xp: ModuleType, device: Device):
        x = xp.asarray([1, 2, 3], device=device)
        assert get_device(cov(x)) == device

    @pytest.mark.skip_xp_backend(Backend.NUMPY_READONLY, reason="xp=xp")
    def test_xp(self, xp: ModuleType):
        xp_assert_close(
            cov(xp.asarray([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]).T, xp=xp),
            xp.asarray([[1.0, -1.0], [-1.0, 1.0]], dtype=xp.float64),
        )


class TestCreateDiagonal:
    @pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no device kwarg in zeros()")
    def test_1d(self, xp: ModuleType):
        # from np.diag tests
        vals = 100 * xp.arange(5, dtype=xp.float64)
        b = xp.zeros((5, 5), dtype=xp.float64)
        for k in range(5):
            b = at(b)[k, k].set(vals[k])
        xp_assert_equal(create_diagonal(vals), b)
        b = xp.zeros((7, 7), dtype=xp.float64)
        c = xp.asarray(b, copy=True)
        for k in range(5):
            b = at(b)[k, k + 2].set(vals[k])
            c = at(c)[k + 2, k].set(vals[k])
        xp_assert_equal(create_diagonal(vals, offset=2), b)
        xp_assert_equal(create_diagonal(vals, offset=-2), c)

    @pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no device kwarg in zeros()")
    @pytest.mark.parametrize("n", range(1, 10))
    @pytest.mark.parametrize("offset", range(1, 10))
    def test_create_diagonal(self, xp: ModuleType, n: int, offset: int):
        # from scipy._lib tests
        rng = np.random.default_rng(2347823)
        one = xp.asarray(1.0)
        x = rng.random(n)
        A = create_diagonal(xp.asarray(x, dtype=one.dtype), offset=offset)
        B = xp.asarray(np.diag(x, offset), dtype=one.dtype)
        xp_assert_equal(A, B)

    def test_0d(self, xp: ModuleType):
        with pytest.raises(ValueError, match="1-dimensional"):
            create_diagonal(xp.asarray(1))

    def test_2d(self, xp: ModuleType):
        with pytest.raises(ValueError, match="1-dimensional"):
            create_diagonal(xp.asarray([[1]]))

    @pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no device kwarg in zeros()")
    def test_device(self, xp: ModuleType, device: Device):
        x = xp.asarray([1, 2, 3], device=device)
        assert get_device(create_diagonal(x)) == device

    @pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no device kwarg in zeros()")
    def test_xp(self, xp: ModuleType):
        x = xp.asarray([1, 2])
        y = create_diagonal(x, xp=xp)
        xp_assert_equal(y, xp.asarray([[1, 0], [0, 2]]))


class TestExpandDims:
    @pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no expand_dims")
    @pytest.mark.xfail_xp_backend(Backend.DASK, reason="tuple index out of range")
    @pytest.mark.xfail_xp_backend(Backend.TORCH, reason="tuple index out of range")
    def test_functionality(self, xp: ModuleType):
        def _squeeze_all(b: Array) -> Array:
            """Mimics `np.squeeze(b)`. `xpx.squeeze`?"""
            for axis in range(b.ndim):
                with contextlib.suppress(ValueError):
                    b = xp.squeeze(b, axis=axis)
            return b

        s = (2, 3, 4, 5)
        a = xp.empty(s)
        for axis in range(-5, 4):
            b = expand_dims(a, axis=axis)
            assert b.shape[axis] == 1
            assert _squeeze_all(b).shape == s

    @pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no expand_dims")
    def test_axis_tuple(self, xp: ModuleType):
        a = xp.empty((3, 3, 3))
        assert expand_dims(a, axis=(0, 1, 2)).shape == (1, 1, 1, 3, 3, 3)
        assert expand_dims(a, axis=(0, -1, -2)).shape == (1, 3, 3, 3, 1, 1)
        assert expand_dims(a, axis=(0, 3, 5)).shape == (1, 3, 3, 1, 3, 1)
        assert expand_dims(a, axis=(0, -3, -5)).shape == (1, 1, 3, 1, 3, 3)

    def test_axis_out_of_range(self, xp: ModuleType):
        s = (2, 3, 4, 5)
        a = xp.empty(s)
        with pytest.raises(IndexError, match="out of bounds"):
            expand_dims(a, axis=-6)
        with pytest.raises(IndexError, match="out of bounds"):
            expand_dims(a, axis=5)

        a = xp.empty((3, 3, 3))
        with pytest.raises(IndexError, match="out of bounds"):
            expand_dims(a, axis=(0, -6))
        with pytest.raises(IndexError, match="out of bounds"):
            expand_dims(a, axis=(0, 5))

    def test_repeated_axis(self, xp: ModuleType):
        a = xp.empty((3, 3, 3))
        with pytest.raises(ValueError, match="Duplicate dimensions"):
            expand_dims(a, axis=(1, 1))

    def test_positive_negative_repeated(self, xp: ModuleType):
        # https://github.com/data-apis/array-api/issues/760#issuecomment-1989449817
        a = xp.empty((2, 3, 4, 5))
        with pytest.raises(ValueError, match="Duplicate dimensions"):
            expand_dims(a, axis=(3, -3))

    @pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no expand_dims")
    def test_device(self, xp: ModuleType, device: Device):
        x = xp.asarray([1, 2, 3], device=device)
        assert get_device(expand_dims(x, axis=0)) == device

    @pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no expand_dims")
    def test_xp(self, xp: ModuleType):
        x = xp.asarray([1, 2, 3])
        y = expand_dims(x, axis=(0, 1, 2), xp=xp)
        assert y.shape == (1, 1, 1, 3)


@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no isdtype")
class TestIsClose:
    # FIXME use lazywhere to avoid warnings on inf
    @pytest.mark.filterwarnings("ignore:invalid value encountered")
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (0.0, 0.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (1.0, -1.0),
            (100.0, 101.0),
            (0, 0),
            (1, 1),
            (1, 2),
            (1, -1),
            (1.0 + 1j, 1.0 + 1j),
            (1.0 + 1j, 1.0 - 1j),
            (float("inf"), float("inf")),
            (float("inf"), 100.0),
            (float("inf"), float("-inf")),
            (float("nan"), float("nan")),
            (float("nan"), 0.0),
            (0.0, float("nan")),
            (1e6, 1e6 + 1),  # True - within rtol
            (1e6, 1e6 + 100),  # False - outside rtol
            (1e-6, 1.1e-6),  # False - outside atol
            (1e-7, 1.1e-7),  # True - outside atol
            (1e6 + 0j, 1e6 + 1j),  # True - within rtol
            (1e6 + 0j, 1e6 + 100j),  # False - outside rtol
        ],
    )
    def test_basic(self, a: float, b: float, xp: ModuleType):
        a_xp = xp.asarray(a)
        b_xp = xp.asarray(b)

        xp_assert_equal(isclose(a_xp, b_xp), xp.asarray(np.isclose(a, b)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_xp = xp.asarray(np.arange(10), dtype=a_xp.dtype)
            ar_xp = a_xp * r_xp
            br_xp = b_xp * r_xp
            ar_np = a * np.arange(10)
            br_np = b * np.arange(10)

        xp_assert_equal(isclose(ar_xp, br_xp), xp.asarray(np.isclose(ar_np, br_np)))

    @pytest.mark.parametrize("dtype", ["float32", "int32"])
    def test_broadcast(self, dtype: str, xp: ModuleType):
        dtype = getattr(xp, dtype)
        a = xp.asarray([1, 2, 3], dtype=dtype)
        b = xp.asarray([[1], [5]], dtype=dtype)
        actual = isclose(a, b)
        expect = xp.asarray(
            [[True, False, False], [False, False, False]], dtype=xp.bool
        )

        xp_assert_equal(actual, expect)

    # FIXME use lazywhere to avoid warnings on inf
    @pytest.mark.filterwarnings("ignore:invalid value encountered")
    def test_some_inf(self, xp: ModuleType):
        a = xp.asarray([0.0, 1.0, float("inf"), float("inf"), float("inf")])
        b = xp.asarray([1e-9, 1.0, float("inf"), float("-inf"), 2.0])
        actual = isclose(a, b)
        xp_assert_equal(actual, xp.asarray([True, True, True, False, False]))

    def test_equal_nan(self, xp: ModuleType):
        a = xp.asarray([float("nan"), float("nan"), 1.0])
        b = xp.asarray([float("nan"), 1.0, float("nan")])
        xp_assert_equal(isclose(a, b), xp.asarray([False, False, False]))
        xp_assert_equal(isclose(a, b, equal_nan=True), xp.asarray([True, False, False]))

    @pytest.mark.parametrize("dtype", ["float32", "complex64", "int32"])
    def test_tolerance(self, dtype: str, xp: ModuleType):
        dtype = getattr(xp, dtype)
        a = xp.asarray([100, 100], dtype=dtype)
        b = xp.asarray([101, 102], dtype=dtype)
        xp_assert_equal(isclose(a, b), xp.asarray([False, False]))
        xp_assert_equal(isclose(a, b, atol=1), xp.asarray([True, False]))
        xp_assert_equal(isclose(a, b, rtol=0.01), xp.asarray([True, False]))

        # Attempt to trigger division by 0 in rtol on int dtype
        xp_assert_equal(isclose(a, b, rtol=0), xp.asarray([False, False]))
        xp_assert_equal(isclose(a, b, atol=1, rtol=0), xp.asarray([True, False]))

    @pytest.mark.parametrize("dtype", ["int8", "uint8"])
    def test_tolerance_integer_overflow(self, dtype: str, xp: ModuleType):
        """1/rtol is too large for dtype"""
        a = xp.asarray([100, 100], dtype=getattr(xp, dtype))
        b = xp.asarray([100, 101], dtype=getattr(xp, dtype))
        xp_assert_equal(isclose(a, b), xp.asarray([True, False]))

    def test_very_small_numbers(self, xp: ModuleType):
        a = xp.asarray([1e-9, 1e-9])
        b = xp.asarray([1.0001e-9, 1.00001e-9])
        # Difference is below default atol
        xp_assert_equal(isclose(a, b), xp.asarray([True, True]))
        # Use only rtol
        xp_assert_equal(isclose(a, b, atol=0), xp.asarray([False, True]))
        xp_assert_equal(isclose(a, b, atol=0, rtol=0), xp.asarray([False, False]))

    def test_bool_dtype(self, xp: ModuleType):
        a = xp.asarray([False, True, False])
        b = xp.asarray([True, True, False])
        xp_assert_equal(isclose(a, b), xp.asarray([False, True, True]))
        xp_assert_equal(isclose(a, b, atol=1), xp.asarray([True, True, True]))
        xp_assert_equal(isclose(a, b, atol=2), xp.asarray([True, True, True]))
        xp_assert_equal(isclose(a, b, rtol=1), xp.asarray([True, True, True]))
        xp_assert_equal(isclose(a, b, rtol=2), xp.asarray([True, True, True]))

        # Test broadcasting
        xp_assert_equal(
            isclose(a, xp.asarray(True), atol=1), xp.asarray([True, True, True])
        )
        xp_assert_equal(
            isclose(xp.asarray(True), b, atol=1), xp.asarray([True, True, True])
        )

    def test_none_shape(self, xp: ModuleType):
        a = xp.asarray([1, 5, 0])
        b = xp.asarray([1, 4, 2])
        b = b[a < 5]
        a = a[a < 5]
        xp_assert_equal(isclose(a, b), xp.asarray([True, False]))

    def test_none_shape_bool(self, xp: ModuleType):
        a = xp.asarray([True, True, False])
        b = xp.asarray([True, False, True])
        b = b[a]
        a = a[a]
        xp_assert_equal(isclose(a, b), xp.asarray([True, False]))

    @pytest.mark.skip_xp_backend(Backend.NUMPY_READONLY, reason="xp=xp")
    @pytest.mark.xfail_xp_backend(Backend.TORCH, reason="Array API 2024.12 support")
    def test_python_scalar(self, xp: ModuleType):
        a = xp.asarray([0.0, 0.1], dtype=xp.float32)
        xp_assert_equal(isclose(a, 0.0), xp.asarray([True, False]))
        xp_assert_equal(isclose(0.0, a), xp.asarray([True, False]))

        a = xp.asarray([0, 1], dtype=xp.int16)
        xp_assert_equal(isclose(a, 0), xp.asarray([True, False]))
        xp_assert_equal(isclose(0, a), xp.asarray([True, False]))

        xp_assert_equal(isclose(0, 0, xp=xp), xp.asarray(True))
        xp_assert_equal(isclose(0, 1, xp=xp), xp.asarray(False))

    def test_all_python_scalars(self):
        with pytest.raises(TypeError, match="Unrecognized"):
            isclose(0, 0)

    def test_xp(self, xp: ModuleType):
        a = xp.asarray([0.0, 0.0])
        b = xp.asarray([1e-9, 1e-4])
        xp_assert_equal(isclose(a, b, xp=xp), xp.asarray([True, False]))


@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no expand_dims")
class TestKron:
    def test_basic(self, xp: ModuleType):
        # Using 0-dimensional array
        a = xp.asarray(1)
        b = xp.asarray([[1, 2], [3, 4]])
        xp_assert_equal(kron(a, b), b)
        xp_assert_equal(kron(b, a), b)

        # Using 1-dimensional array
        a = xp.asarray([3])
        b = xp.asarray([[1, 2], [3, 4]])
        k = xp.asarray([[3, 6], [9, 12]])
        xp_assert_equal(kron(a, b), k)
        xp_assert_equal(kron(b, a), k)

        # Using 3-dimensional array
        a = xp.asarray([[[1]], [[2]]])
        b = xp.asarray([[1, 2], [3, 4]])
        k = xp.asarray([[[1, 2], [3, 4]], [[2, 4], [6, 8]]])
        xp_assert_equal(kron(a, b), k)
        xp_assert_equal(kron(b, a), k)

    def test_kron_smoke(self, xp: ModuleType):
        a = xp.ones((3, 3))
        b = xp.ones((3, 3))
        k = xp.ones((9, 9))

        xp_assert_equal(kron(a, b), k)

    @pytest.mark.parametrize(
        ("shape_a", "shape_b"),
        [
            ((1, 1), (1, 1)),
            ((1, 2, 3), (4, 5, 6)),
            ((2, 2), (2, 2, 2)),
            ((1, 0), (1, 1)),
            ((2, 0, 2), (2, 2)),
            ((2, 0, 0, 2), (2, 0, 2)),
        ],
    )
    def test_kron_shape(
        self, xp: ModuleType, shape_a: tuple[int, ...], shape_b: tuple[int, ...]
    ):
        a = xp.ones(shape_a)
        b = xp.ones(shape_b)
        normalised_shape_a = xp.asarray(
            (1,) * max(0, len(shape_b) - len(shape_a)) + shape_a
        )
        normalised_shape_b = xp.asarray(
            (1,) * max(0, len(shape_a) - len(shape_b)) + shape_b
        )
        expected_shape = tuple(
            int(dim) for dim in xp.multiply(normalised_shape_a, normalised_shape_b)
        )

        k = kron(a, b)
        assert k.shape == expected_shape

    def test_python_scalar(self, xp: ModuleType):
        a = 1
        # Test no dtype promotion to xp.asarray(a); use b.dtype
        b = xp.asarray([[1, 2], [3, 4]], dtype=xp.int16)
        xp_assert_equal(kron(a, b), b)
        xp_assert_equal(kron(b, a), b)
        xp_assert_equal(kron(1, 1, xp=xp), xp.asarray(1))

    def test_all_python_scalars(self):
        with pytest.raises(TypeError, match="Unrecognized"):
            kron(1, 1)

    def test_device(self, xp: ModuleType, device: Device):
        x1 = xp.asarray([1, 2, 3], device=device)
        x2 = xp.asarray([4, 5], device=device)
        assert get_device(kron(x1, x2)) == device

    def test_xp(self, xp: ModuleType):
        a = xp.ones((3, 3))
        b = xp.ones((3, 3))
        k = xp.ones((9, 9))
        xp_assert_equal(kron(a, b, xp=xp), k)


class TestNUnique:
    def test_simple(self, xp: ModuleType):
        a = xp.asarray([[1, 1], [0, 2], [2, 2]])
        xp_assert_equal(nunique(a), xp.asarray(3))

    def test_empty(self, xp: ModuleType):
        a = xp.asarray([])
        xp_assert_equal(nunique(a), xp.asarray(0))

    def test_device(self, xp: ModuleType, device: Device):
        a = xp.asarray(0.0, device=device)
        assert get_device(nunique(a)) == device

    def test_xp(self, xp: ModuleType):
        a = xp.asarray([[1, 1], [0, 2], [2, 2]])
        xp_assert_equal(nunique(a, xp=xp), xp.asarray(3))


@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no arange, no device")
class TestPad:
    def test_simple(self, xp: ModuleType):
        a = xp.arange(1, 4)
        padded = pad(a, 2)
        xp_assert_equal(padded, xp.asarray([0, 0, 1, 2, 3, 0, 0]))

    def test_fill_value(self, xp: ModuleType):
        a = xp.arange(1, 4)
        padded = pad(a, 2, constant_values=42)
        xp_assert_equal(padded, xp.asarray([42, 42, 1, 2, 3, 42, 42]))

    def test_ndim(self, xp: ModuleType):
        a = xp.reshape(xp.arange(2 * 3 * 4), (2, 3, 4))
        padded = pad(a, 2)
        assert padded.shape == (6, 7, 8)

    def test_mode_not_implemented(self, xp: ModuleType):
        a = xp.arange(3)
        with pytest.raises(NotImplementedError, match="Only `'constant'`"):
            pad(a, 2, mode="edge")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

    def test_device(self, xp: ModuleType, device: Device):
        a = xp.asarray(0.0, device=device)
        assert get_device(pad(a, 2)) == device

    def test_xp(self, xp: ModuleType):
        padded = pad(xp.asarray(0), 1, xp=xp)
        xp_assert_equal(padded, xp.asarray(0))

    def test_tuple_width(self, xp: ModuleType):
        a = xp.reshape(xp.arange(12), (3, 4))
        padded = pad(a, (1, 0))
        assert padded.shape == (4, 5)

        padded = pad(a, (1, 2))
        assert padded.shape == (6, 7)

        with pytest.raises((ValueError, RuntimeError)):
            pad(a, [(1, 2, 3)])  # type: ignore[list-item]  # pyright: ignore[reportArgumentType]

    def test_sequence_of_tuples_width(self, xp: ModuleType):
        a = xp.reshape(xp.arange(12), (3, 4))

        padded = pad(a, ((1, 0), (0, 2)))
        assert padded.shape == (4, 6)
        padded = pad(a, ((1, 0), (0, 0)))
        assert padded.shape == (4, 4)


assume_unique = pytest.mark.parametrize(
    "assume_unique",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail_xp_backend(
                Backend.DASK, reason="NaN-shaped arrays"
            ),
        ),
    ],
)


@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no device kwarg in asarray()")
class TestSetDiff1D:
    @pytest.mark.xfail_xp_backend(Backend.DASK, reason="NaN-shaped arrays")
    @pytest.mark.xfail_xp_backend(
        Backend.TORCH, reason="index_select not implemented for uint32"
    )
    def test_setdiff1d(self, xp: ModuleType):
        x1 = xp.asarray([6, 5, 4, 7, 1, 2, 7, 4])
        x2 = xp.asarray([2, 4, 3, 3, 2, 1, 5])

        expected = xp.asarray([6, 7])
        actual = setdiff1d(x1, x2)
        xp_assert_equal(actual, expected)

        x1 = xp.arange(21)
        x2 = xp.arange(19)
        expected = xp.asarray([19, 20])
        actual = setdiff1d(x1, x2)
        xp_assert_equal(actual, expected)

        xp_assert_equal(setdiff1d(xp.empty(0), xp.empty(0)), xp.empty(0))
        x1 = xp.empty(0, dtype=xp.uint32)
        x2 = x1
        assert xp.isdtype(setdiff1d(x1, x2).dtype, xp.uint32)

    def test_assume_unique(self, xp: ModuleType):
        x1 = xp.asarray([3, 2, 1])
        x2 = xp.asarray([7, 5, 2])
        expected = xp.asarray([3, 1])
        actual = setdiff1d(x1, x2, assume_unique=True)
        xp_assert_equal(actual, expected)

    @assume_unique
    @pytest.mark.parametrize("shape1", [(), (1,), (1, 1)])
    @pytest.mark.parametrize("shape2", [(), (1,), (1, 1)])
    def test_shapes(
        self,
        assume_unique: bool,
        shape1: tuple[int, ...],
        shape2: tuple[int, ...],
        xp: ModuleType,
    ):
        x1 = xp.zeros(shape1)
        x2 = xp.zeros(shape2)
        actual = setdiff1d(x1, x2, assume_unique=assume_unique)
        xp_assert_equal(actual, xp.empty((0,)))

    @assume_unique
    @pytest.mark.skip_xp_backend(Backend.NUMPY_READONLY, reason="xp=xp")
    def test_python_scalar(self, xp: ModuleType, assume_unique: bool):
        # Test no dtype promotion to xp.asarray(x2); use x1.dtype
        x1 = xp.asarray([3, 1, 2], dtype=xp.int16)
        x2 = 3
        actual = setdiff1d(x1, x2, assume_unique=assume_unique)
        xp_assert_equal(actual, xp.asarray([1, 2], dtype=xp.int16))

        actual = setdiff1d(x2, x1, assume_unique=assume_unique)
        xp_assert_equal(actual, xp.asarray([], dtype=xp.int16))

        xp_assert_equal(
            setdiff1d(0, 0, assume_unique=assume_unique, xp=xp),
            xp.asarray([0])[:0],  # Default int dtype for backend
        )

    @pytest.mark.parametrize("assume_unique", [True, False])
    def test_all_python_scalars(self, assume_unique: bool):
        with pytest.raises(TypeError, match="Unrecognized"):
            setdiff1d(0, 0, assume_unique=assume_unique)

    @assume_unique
    def test_device(self, xp: ModuleType, device: Device, assume_unique: bool):
        x1 = xp.asarray([3, 8, 20], device=device)
        x2 = xp.asarray([2, 3, 4], device=device)
        assert get_device(setdiff1d(x1, x2, assume_unique=assume_unique)) == device

    @pytest.mark.skip_xp_backend(Backend.NUMPY_READONLY, reason="xp=xp")
    def test_xp(self, xp: ModuleType):
        x1 = xp.asarray([3, 8, 20])
        x2 = xp.asarray([2, 3, 4])
        expected = xp.asarray([8, 20])
        actual = setdiff1d(x1, x2, assume_unique=True, xp=xp)
        xp_assert_equal(actual, expected)


@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no isdtype")
class TestSinc:
    def test_simple(self, xp: ModuleType):
        xp_assert_equal(sinc(xp.asarray(0.0)), xp.asarray(1.0))
        w = sinc(xp.linspace(-1, 1, 100))
        # check symmetry
        xp_assert_close(w, xp.flip(w, axis=0))

    @pytest.mark.parametrize("x", [0, 1 + 3j])
    def test_dtype(self, xp: ModuleType, x: int | complex):
        with pytest.raises(ValueError, match="real floating data type"):
            sinc(xp.asarray(x))

    def test_3d(self, xp: ModuleType):
        x = xp.reshape(xp.arange(18, dtype=xp.float64), (3, 3, 2))
        expected = xp.zeros((3, 3, 2), dtype=xp.float64)
        expected = at(expected)[0, 0, 0].set(1.0)
        xp_assert_close(sinc(x), expected, atol=1e-15)

    def test_device(self, xp: ModuleType, device: Device):
        x = xp.asarray(0.0, device=device)
        assert get_device(sinc(x)) == device

    def test_xp(self, xp: ModuleType):
        xp_assert_equal(sinc(xp.asarray(0.0), xp=xp), xp.asarray(1.0))
