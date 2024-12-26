import contextlib
import warnings

# data-apis/array-api-strict#6
import array_api_strict as xp  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

from array_api_extra import (
    atleast_nd,
    cov,
    create_diagonal,
    expand_dims,
    kron,
    pad,
    setdiff1d,
    sinc,
)
from array_api_extra._lib._typing import Array


class TestAtLeastND:
    def test_0D(self):
        x = xp.asarray(1)

        y = atleast_nd(x, ndim=0)
        assert_array_equal(y, x)

        y = atleast_nd(x, ndim=1)
        assert_array_equal(y, xp.ones((1,)))

        y = atleast_nd(x, ndim=5)
        assert_array_equal(y, xp.ones((1, 1, 1, 1, 1)))

    def test_1D(self):
        x = xp.asarray([0, 1])

        y = atleast_nd(x, ndim=0)
        assert_array_equal(y, x)

        y = atleast_nd(x, ndim=1)
        assert_array_equal(y, x)

        y = atleast_nd(x, ndim=2)
        assert_array_equal(y, xp.asarray([[0, 1]]))

        y = atleast_nd(x, ndim=5)
        assert_array_equal(y, xp.reshape(xp.arange(2), (1, 1, 1, 1, 2)))

    def test_2D(self):
        x = xp.asarray([[3]])

        y = atleast_nd(x, ndim=0)
        assert_array_equal(y, x)

        y = atleast_nd(x, ndim=2)
        assert_array_equal(y, x)

        y = atleast_nd(x, ndim=3)
        assert_array_equal(y, 3 * xp.ones((1, 1, 1)))

        y = atleast_nd(x, ndim=5)
        assert_array_equal(y, 3 * xp.ones((1, 1, 1, 1, 1)))

    def test_5D(self):
        x = xp.ones((1, 1, 1, 1, 1))

        y = atleast_nd(x, ndim=0)
        assert_array_equal(y, x)

        y = atleast_nd(x, ndim=4)
        assert_array_equal(y, x)

        y = atleast_nd(x, ndim=5)
        assert_array_equal(y, x)

        y = atleast_nd(x, ndim=6)
        assert_array_equal(y, xp.ones((1, 1, 1, 1, 1, 1)))

        y = atleast_nd(x, ndim=9)
        assert_array_equal(y, xp.ones((1, 1, 1, 1, 1, 1, 1, 1, 1)))

    def test_device(self):
        device = xp.Device("device1")
        x = xp.asarray([1, 2, 3], device=device)
        assert atleast_nd(x, ndim=2).device == device

    def test_xp(self):
        x = xp.asarray(1)
        y = atleast_nd(x, ndim=0, xp=xp)
        assert_array_equal(y, x)


class TestCov:
    def test_basic(self):
        assert_allclose(
            cov(xp.asarray([[0, 2], [1, 1], [2, 0]]).T),
            xp.asarray([[1.0, -1.0], [-1.0, 1.0]]),
        )

    def test_complex(self):
        x = xp.asarray([[1, 2, 3], [1j, 2j, 3j]])
        res = xp.asarray([[1.0, -1.0j], [1.0j, 1.0]])
        assert_allclose(cov(x), res)

    def test_empty(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", RuntimeWarning)
            assert_array_equal(cov(xp.asarray([])), xp.nan)
            assert_array_equal(
                cov(xp.reshape(xp.asarray([]), (0, 2))),
                xp.reshape(xp.asarray([]), (0, 0)),
            )
            assert_array_equal(
                cov(xp.reshape(xp.asarray([]), (2, 0))),
                xp.asarray([[xp.nan, xp.nan], [xp.nan, xp.nan]]),
            )

    def test_combination(self):
        x = xp.asarray([-2.1, -1, 4.3])
        y = xp.asarray([3, 1.1, 0.12])
        X = xp.stack((x, y), axis=0)
        desired = xp.asarray([[11.71, -4.286], [-4.286, 2.144133]])
        assert_allclose(cov(X), desired, rtol=1e-6)
        assert_allclose(cov(x), xp.asarray(11.71))
        assert_allclose(cov(y), xp.asarray(2.144133), rtol=1e-6)

    def test_device(self):
        device = xp.Device("device1")
        x = xp.asarray([1, 2, 3], device=device)
        assert cov(x).device == device

    def test_xp(self):
        assert_allclose(
            cov(xp.asarray([[0, 2], [1, 1], [2, 0]]).T, xp=xp),
            xp.asarray([[1.0, -1.0], [-1.0, 1.0]]),
        )


class TestCreateDiagonal:
    def test_1d(self):
        # from np.diag tests
        vals = 100 * xp.arange(5, dtype=xp.float64)
        b = xp.zeros((5, 5))
        for k in range(5):
            b[k, k] = vals[k]
        assert_array_equal(create_diagonal(vals), b)
        b = xp.zeros((7, 7))
        c = xp.asarray(b, copy=True)
        for k in range(5):
            b[k, k + 2] = vals[k]
            c[k + 2, k] = vals[k]
        assert_array_equal(create_diagonal(vals, offset=2), b)
        assert_array_equal(create_diagonal(vals, offset=-2), c)

    @pytest.mark.parametrize("n", range(1, 10))
    @pytest.mark.parametrize("offset", range(1, 10))
    def test_create_diagonal(self, n: int, offset: int):
        # from scipy._lib tests
        rng = np.random.default_rng(2347823)
        one = xp.asarray(1.0)
        x = rng.random(n)
        A = create_diagonal(xp.asarray(x, dtype=one.dtype), offset=offset)
        B = xp.asarray(np.diag(x, offset), dtype=one.dtype)
        assert_array_equal(A, B)

    def test_0d(self):
        with pytest.raises(ValueError, match="1-dimensional"):
            create_diagonal(xp.asarray(1))

    def test_2d(self):
        with pytest.raises(ValueError, match="1-dimensional"):
            create_diagonal(xp.asarray([[1]]))

    def test_device(self):
        device = xp.Device("device1")
        x = xp.asarray([1, 2, 3], device=device)
        assert create_diagonal(x).device == device

    def test_xp(self):
        x = xp.asarray([1, 2])
        y = create_diagonal(x, xp=xp)
        assert_array_equal(y, xp.asarray([[1, 0], [0, 2]]))


class TestExpandDims:
    def test_functionality(self):
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

    def test_axis_tuple(self):
        a = xp.empty((3, 3, 3))
        assert expand_dims(a, axis=(0, 1, 2)).shape == (1, 1, 1, 3, 3, 3)
        assert expand_dims(a, axis=(0, -1, -2)).shape == (1, 3, 3, 3, 1, 1)
        assert expand_dims(a, axis=(0, 3, 5)).shape == (1, 3, 3, 1, 3, 1)
        assert expand_dims(a, axis=(0, -3, -5)).shape == (1, 1, 3, 1, 3, 3)

    def test_axis_out_of_range(self):
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

    def test_repeated_axis(self):
        a = xp.empty((3, 3, 3))
        with pytest.raises(ValueError, match="Duplicate dimensions"):
            expand_dims(a, axis=(1, 1))

    def test_positive_negative_repeated(self):
        # https://github.com/data-apis/array-api/issues/760#issuecomment-1989449817
        a = xp.empty((2, 3, 4, 5))
        with pytest.raises(ValueError, match="Duplicate dimensions"):
            expand_dims(a, axis=(3, -3))

    def test_device(self):
        device = xp.Device("device1")
        x = xp.asarray([1, 2, 3], device=device)
        assert expand_dims(x, axis=0).device == device

    def test_xp(self):
        x = xp.asarray([1, 2, 3])
        y = expand_dims(x, axis=(0, 1, 2), xp=xp)
        assert y.shape == (1, 1, 1, 3)


class TestKron:
    def test_basic(self):
        # Using 0-dimensional array
        a = xp.asarray(1)
        b = xp.asarray([[1, 2], [3, 4]])
        k = xp.asarray([[1, 2], [3, 4]])
        assert_array_equal(kron(a, b), k)
        a = xp.asarray([[1, 2], [3, 4]])
        b = xp.asarray(1)
        assert_array_equal(kron(a, b), k)

        # Using 1-dimensional array
        a = xp.asarray([3])
        b = xp.asarray([[1, 2], [3, 4]])
        k = xp.asarray([[3, 6], [9, 12]])
        assert_array_equal(kron(a, b), k)
        a = xp.asarray([[1, 2], [3, 4]])
        b = xp.asarray([3])
        assert_array_equal(kron(a, b), k)

        # Using 3-dimensional array
        a = xp.asarray([[[1]], [[2]]])
        b = xp.asarray([[1, 2], [3, 4]])
        k = xp.asarray([[[1, 2], [3, 4]], [[2, 4], [6, 8]]])
        assert_array_equal(kron(a, b), k)
        a = xp.asarray([[1, 2], [3, 4]])
        b = xp.asarray([[[1]], [[2]]])
        k = xp.asarray([[[1, 2], [3, 4]], [[2, 4], [6, 8]]])
        assert_array_equal(kron(a, b), k)

    def test_kron_smoke(self):
        a = xp.ones((3, 3))
        b = xp.ones((3, 3))
        k = xp.ones((9, 9))

        assert_array_equal(kron(a, b), k)

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
    def test_kron_shape(self, shape_a: tuple[int, ...], shape_b: tuple[int, ...]):
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
        assert_equal(k.shape, expected_shape, err_msg="Unexpected shape from kron")

    def test_device(self):
        device = xp.Device("device1")
        x1 = xp.asarray([1, 2, 3], device=device)
        x2 = xp.asarray([4, 5], device=device)
        assert kron(x1, x2).device == device

    def test_xp(self):
        a = xp.ones((3, 3))
        b = xp.ones((3, 3))
        k = xp.ones((9, 9))
        assert_array_equal(kron(a, b, xp=xp), k)


class TestSetDiff1D:
    def test_setdiff1d(self):
        x1 = xp.asarray([6, 5, 4, 7, 1, 2, 7, 4])
        x2 = xp.asarray([2, 4, 3, 3, 2, 1, 5])

        expected = xp.asarray([6, 7])
        actual = setdiff1d(x1, x2)
        assert_array_equal(actual, expected)

        x1 = xp.arange(21)
        x2 = xp.arange(19)
        expected = xp.asarray([19, 20])
        actual = setdiff1d(x1, x2)
        assert_array_equal(actual, expected)

        assert_array_equal(setdiff1d(xp.empty(0), xp.empty(0)), xp.empty(0))
        x1 = xp.empty(0, dtype=xp.uint32)
        x2 = x1
        assert_equal(setdiff1d(x1, x2).dtype, xp.uint32)

    def test_assume_unique(self):
        x1 = xp.asarray([3, 2, 1])
        x2 = xp.asarray([7, 5, 2])
        expected = xp.asarray([3, 1])
        actual = setdiff1d(x1, x2, assume_unique=True)
        assert_array_equal(actual, expected)

    def test_device(self):
        device = xp.Device("device1")
        x1 = xp.asarray([3, 8, 20], device=device)
        x2 = xp.asarray([2, 3, 4], device=device)
        assert setdiff1d(x1, x2).device == device

    def test_xp(self):
        x1 = xp.asarray([3, 8, 20])
        x2 = xp.asarray([2, 3, 4])
        expected = xp.asarray([8, 20])
        actual = setdiff1d(x1, x2, xp=xp)
        assert_array_equal(actual, expected)


class TestSinc:
    def test_simple(self):
        assert_array_equal(sinc(xp.asarray(0.0)), xp.asarray(1.0))
        w = sinc(xp.linspace(-1, 1, 100))
        # check symmetry
        assert_allclose(w, xp.flip(w, axis=0))

    @pytest.mark.parametrize("x", [0, 1 + 3j])
    def test_dtype(self, x: int | complex):
        with pytest.raises(ValueError, match="real floating data type"):
            sinc(xp.asarray(x))

    def test_3d(self):
        x = xp.reshape(xp.arange(18, dtype=xp.float64), (3, 3, 2))
        expected = xp.zeros((3, 3, 2))
        expected[0, 0, 0] = 1.0
        assert_allclose(sinc(x), expected, atol=1e-15)

    def test_device(self):
        device = xp.Device("device1")
        x = xp.asarray(0.0, device=device)
        assert sinc(x).device == device

    def test_xp(self):
        assert_array_equal(sinc(xp.asarray(0.0), xp=xp), xp.asarray(1.0))


class TestPad:
    def test_simple(self):
        a = xp.arange(1, 4)
        padded = pad(a, 2)
        assert xp.all(padded == xp.asarray([0, 0, 1, 2, 3, 0, 0]))

    def test_fill_value(self):
        a = xp.arange(1, 4)
        padded = pad(a, 2, constant_values=42)
        assert xp.all(padded == xp.asarray([42, 42, 1, 2, 3, 42, 42]))

    def test_ndim(self):
        a = xp.reshape(xp.arange(2 * 3 * 4), (2, 3, 4))
        padded = pad(a, 2)
        assert padded.shape == (6, 7, 8)

    def test_mode_not_implemented(self):
        a = xp.arange(3)
        with pytest.raises(NotImplementedError, match="Only `'constant'`"):
            pad(a, 2, mode="edge")

    def test_device(self):
        device = xp.Device("device1")
        a = xp.asarray(0.0, device=device)
        assert pad(a, 2).device == device

    def test_xp(self):
        assert_array_equal(pad(xp.asarray(0), 1, xp=xp), xp.zeros(3))
