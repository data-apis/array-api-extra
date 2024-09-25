from __future__ import annotations

import warnings

# array-api-strict#6
import array_api_strict as xp  # type: ignore[import-untyped]
from numpy.testing import assert_allclose, assert_array_equal

from array_api_extra import atleast_nd, cov


class TestAtLeastND:
    def test_0D(self):
        x = xp.asarray(1)

        y = atleast_nd(x, ndim=0, xp=xp)
        assert_array_equal(y, x)

        y = atleast_nd(x, ndim=1, xp=xp)
        assert_array_equal(y, xp.ones((1,)))

        y = atleast_nd(x, ndim=5, xp=xp)
        assert_array_equal(y, xp.ones((1, 1, 1, 1, 1)))

    def test_1D(self):
        x = xp.asarray([0, 1])

        y = atleast_nd(x, ndim=0, xp=xp)
        assert_array_equal(y, x)

        y = atleast_nd(x, ndim=1, xp=xp)
        assert_array_equal(y, x)

        y = atleast_nd(x, ndim=2, xp=xp)
        assert_array_equal(y, xp.asarray([[0, 1]]))

        y = atleast_nd(x, ndim=5, xp=xp)
        assert_array_equal(y, xp.reshape(xp.arange(2), (1, 1, 1, 1, 2)))

    def test_2D(self):
        x = xp.asarray([[3]])

        y = atleast_nd(x, ndim=0, xp=xp)
        assert_array_equal(y, x)

        y = atleast_nd(x, ndim=2, xp=xp)
        assert_array_equal(y, x)

        y = atleast_nd(x, ndim=3, xp=xp)
        assert_array_equal(y, 3 * xp.ones((1, 1, 1)))

        y = atleast_nd(x, ndim=5, xp=xp)
        assert_array_equal(y, 3 * xp.ones((1, 1, 1, 1, 1)))

    def test_5D(self):
        x = xp.ones((1, 1, 1, 1, 1))

        y = atleast_nd(x, ndim=0, xp=xp)
        assert_array_equal(y, x)

        y = atleast_nd(x, ndim=4, xp=xp)
        assert_array_equal(y, x)

        y = atleast_nd(x, ndim=5, xp=xp)
        assert_array_equal(y, x)

        y = atleast_nd(x, ndim=6, xp=xp)
        assert_array_equal(y, xp.ones((1, 1, 1, 1, 1, 1)))

        y = atleast_nd(x, ndim=9, xp=xp)
        assert_array_equal(y, xp.ones((1, 1, 1, 1, 1, 1, 1, 1, 1)))


class TestCov:
    def test_basic(self):
        assert_allclose(
            cov(xp.asarray([[0, 2], [1, 1], [2, 0]]).T, xp=xp),
            xp.asarray([[1.0, -1.0], [-1.0, 1.0]]),
        )

    def test_complex(self):
        x = xp.asarray([[1, 2, 3], [1j, 2j, 3j]])
        res = xp.asarray([[1.0, -1.0j], [1.0j, 1.0]])
        assert_allclose(cov(x, xp=xp), res)

    def test_empty(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", RuntimeWarning)
            assert_array_equal(cov(xp.asarray([]), xp=xp), xp.nan)
            assert_array_equal(
                cov(xp.reshape(xp.asarray([]), (0, 2)), xp=xp),
                xp.reshape(xp.asarray([]), (0, 0)),
            )
            assert_array_equal(
                cov(xp.reshape(xp.asarray([]), (2, 0)), xp=xp),
                xp.asarray([[xp.nan, xp.nan], [xp.nan, xp.nan]]),
            )
