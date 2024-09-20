from __future__ import annotations

import array_api_strict as xp  # type: ignore[import-not-found]

from array_api_extra import atleast_nd


class TestAtLeastND:
    def test_1d_to_2d(self):
        x = xp.asarray([0, 1])
        y = atleast_nd(x, ndim=2, xp=xp)
        assert y.ndim == 2
