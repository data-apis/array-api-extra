from __future__ import annotations

import array_api_strict as xp  # type: ignore[import-untyped]
from numpy.testing import assert_array_equal

from array_api_extra import atleast_nd  # type: ignore[import-not-found]


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
