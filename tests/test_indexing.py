from collections.abc import Callable

import numpy as np
import pytest

from array_api_extra import diag_indices, tril_indices, triu_indices, unravel_index
from array_api_extra._lib._backends import Backend
from array_api_extra._lib._compat import device as get_device
from array_api_extra._lib._typing import Array, ArrayNamespace, Device
from array_api_extra.testing import assert_equal, lazy_xp_function

lazy_xp_function(diag_indices)
lazy_xp_function(tril_indices)
lazy_xp_function(triu_indices)


@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no arange", strict=False)
class TestDiagIndices:
    def test_basic(self, xp: ArrayNamespace):
        rows, cols = diag_indices(5, xp=xp)
        ref_rows, ref_cols = np.diag_indices(5)
        assert_equal(rows, xp.asarray(ref_rows))
        assert_equal(cols, xp.asarray(ref_cols))

    @pytest.mark.parametrize("n", [2, 4, 7])
    @pytest.mark.parametrize("ndim", [1, 2, 3, 4])
    def test_ndim(self, xp: ArrayNamespace, n: int, ndim: int):
        idx = diag_indices(n, ndim=ndim, xp=xp)
        assert len(idx) == ndim
        ref = np.diag_indices(n, ndim=ndim)
        for got, expected in zip(idx, ref, strict=True):
            assert_equal(got, xp.asarray(expected))

    def test_empty(self, xp: ArrayNamespace):
        rows, cols = diag_indices(0, xp=xp)
        assert rows.shape == (0,)
        assert cols.shape == (0,)

    def test_validation(self, xp: ArrayNamespace):
        with pytest.raises(ValueError, match="`n` must be non-negative"):
            _ = diag_indices(-1, xp=xp)
        with pytest.raises(ValueError, match="`ndim` must be >= 1"):
            _ = diag_indices(3, ndim=0, xp=xp)

    def test_device(self, xp: ArrayNamespace, device: Device):
        default_device = get_device(xp.empty(0))
        rows, cols = diag_indices(3, device=None, xp=xp)
        assert get_device(rows) == default_device
        assert get_device(cols) == default_device
        rows, cols = diag_indices(3, device=device, xp=xp)
        assert get_device(rows) == device
        assert get_device(cols) == device


@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no arange/nonzero", strict=False)
@pytest.mark.xfail_xp_backend(
    Backend.ARRAY_API_STRICTEST,
    reason="generic path uses nonzero (data-dependent)",
    strict=False,
)
@pytest.mark.parametrize(
    ("xpx_fn", "np_fn"),
    [(tril_indices, np.tril_indices), (triu_indices, np.triu_indices)],
    ids=["tril", "triu"],
)
class TestTriIndices:
    def test_basic(
        self,
        xp: ArrayNamespace,
        xpx_fn: Callable[..., tuple[Array, Array]],
        np_fn: Callable[..., tuple[Array, Array]],
    ):
        rows, cols = xpx_fn(4, xp=xp)
        ref_rows, ref_cols = np_fn(4)
        assert_equal(rows, xp.asarray(ref_rows))
        assert_equal(cols, xp.asarray(ref_cols))

    @pytest.mark.parametrize("offset", [-2, -1, 0, 1, 2])
    def test_offset(
        self,
        xp: ArrayNamespace,
        xpx_fn: Callable[..., tuple[Array, Array]],
        np_fn: Callable[..., tuple[Array, Array]],
        offset: int,
    ):
        rows, cols = xpx_fn(5, offset=offset, xp=xp)
        ref_rows, ref_cols = np_fn(5, k=offset)
        assert_equal(rows, xp.asarray(ref_rows))
        assert_equal(cols, xp.asarray(ref_cols))

    def test_rectangular(
        self,
        xp: ArrayNamespace,
        xpx_fn: Callable[..., tuple[Array, Array]],
        np_fn: Callable[..., tuple[Array, Array]],
    ):
        rows, cols = xpx_fn(3, m=5, xp=xp)
        ref_rows, ref_cols = np_fn(3, m=5)
        assert_equal(rows, xp.asarray(ref_rows))
        assert_equal(cols, xp.asarray(ref_cols))

    @pytest.mark.xfail_xp_backend(
        Backend.DASK, reason="dask: no 2D fancy indexing", strict=False
    )
    def test_use_to_read(
        self,
        xp: ArrayNamespace,
        xpx_fn: Callable[..., tuple[Array, Array]],
        np_fn: Callable[..., tuple[Array, Array]],
    ):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 100, (4, 4))
        a_xp = xp.asarray(a)
        rows, cols = xpx_fn(4, xp=xp)
        assert_equal(a_xp[rows, cols], xp.asarray(a[np_fn(4)]))

    def test_validation(
        self,
        xp: ArrayNamespace,
        xpx_fn: Callable[..., tuple[Array, Array]],
        np_fn: Callable[..., tuple[Array, Array]],  # noqa: ARG002  # pytest param
    ):
        with pytest.raises(ValueError, match="`n` must be non-negative"):
            _ = xpx_fn(-1, xp=xp)
        with pytest.raises(ValueError, match="`m` must be non-negative"):
            _ = xpx_fn(3, m=-1, xp=xp)

    def test_device(
        self,
        xp: ArrayNamespace,
        device: Device,
        xpx_fn: Callable[..., tuple[Array, Array]],
        np_fn: Callable[..., tuple[Array, Array]],  # noqa: ARG002  # pytest param
    ):
        default_device = get_device(xp.empty(0))
        rows, cols = xpx_fn(4, device=None, xp=xp)
        assert get_device(rows) == default_device
        assert get_device(cols) == default_device
        rows, cols = xpx_fn(4, device=device, xp=xp)
        assert get_device(rows) == device
        assert get_device(cols) == device


class TestUnravelIndex:
    def test_simple(self, xp: ArrayNamespace):
        indices = xp.asarray([22, 41, 37])
        shape = (7, 6)
        expected = (xp.asarray([3, 6, 6]), xp.asarray([4, 5, 1]))
        res = unravel_index(indices, shape)
        for res_arr, exp_arr in zip(res, expected, strict=True):
            assert_equal(res_arr, exp_arr)

        indices = xp.asarray([0, 1, 2, 3, 4, 5])
        shape = (3, 2)
        expected = (
            xp.asarray([0, 0, 1, 1, 2, 2]),
            xp.asarray([0, 1, 0, 1, 0, 1]),
        )
        res = unravel_index(indices, shape)
        for res_arr, exp_arr in zip(res, expected, strict=True):
            assert_equal(res_arr, exp_arr)

    def test_indices_scalar(self, xp: ArrayNamespace):
        indices = xp.asarray(1621)
        shape = (6, 7, 8, 9)
        expected = (xp.asarray(3), xp.asarray(1), xp.asarray(4), xp.asarray(1))
        res = unravel_index(indices, shape)
        # a tuple of integers is expected
        assert res == expected

    def test_indices_2d(self, xp: ArrayNamespace):
        indices = xp.asarray([[1234], [5678]])
        shape = (10, 10, 10, 10)
        expected = (
            xp.asarray([[1], [5]]),
            xp.asarray([[2], [6]]),
            xp.asarray([[3], [7]]),
            xp.asarray([[4], [8]]),
        )
        res = unravel_index(indices, shape)
        for res_arr, exp_arr in zip(res, expected, strict=True):
            assert_equal(res_arr, exp_arr)

    def test_device(self, xp: ArrayNamespace, device: Device):
        indices = xp.asarray([4, 1], device=device)
        shape = (3, 2)
        res = unravel_index(indices, shape)
        for res_arr in res:
            assert get_device(res_arr) == device

    @pytest.mark.skip_xp_backend(Backend.NUMPY_READONLY, reason="xp=xp")
    def test_xp(self, xp: ArrayNamespace):
        indices = xp.asarray([1, 5])
        shape = (3, 2)
        expected = (
            xp.asarray([0, 2]),
            xp.asarray([1, 1]),
        )
        res = unravel_index(indices, shape, xp=xp)
        for res_arr, exp_arr in zip(res, expected, strict=True):
            assert_equal(res_arr, exp_arr)
