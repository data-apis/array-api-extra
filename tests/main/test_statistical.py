import math
import warnings
from typing import Any, cast

import numpy as np
import pytest

from array_api_extra import cov, nanmax, nanmean, nanmin, nansum
from array_api_extra._lib._backends import Backend
from array_api_extra._lib._compat import array_namespace
from array_api_extra._lib._compat import device as get_device
from array_api_extra._lib._typing import Array, ArrayNamespace, Device
from array_api_extra.testing import assert_close, assert_equal, lazy_xp_function

lazy_xp_function(cov)
lazy_xp_function(nanmean)
lazy_xp_function(nansum)


class TestCov:
    def test_basic(self, xp: ArrayNamespace):
        assert_close(
            cov(xp.asarray([[0, 2], [1, 1], [2, 0]], dtype=xp.float64).T),
            xp.asarray([[1.0, -1.0], [-1.0, 1.0]], dtype=xp.float64),
        )

    def test_complex(self, xp: ArrayNamespace):
        actual = cov(xp.asarray([[1, 2, 3], [1j, 2j, 3j]], dtype=xp.complex128))
        expect = xp.asarray([[1.0, -1.0j], [1.0j, 1.0]], dtype=xp.complex128)
        assert_close(actual, expect)

    def test_complex_with_weights(self, xp: ArrayNamespace):
        m = np.asarray(
            [[1 + 1j, 2 + 2j, 4 + 1j], [3 - 1j, 5 + 2j, 7 + 0j]],
            dtype=np.complex128,
        )
        weights = np.asarray([1.0, 2.0, 1.0])
        correction = 0.5  # Force the generic implementation.

        weight_sum = weights.sum()
        avg = (m * weights).sum(axis=-1, keepdims=True) / weight_sum
        centered = m - avg
        normalizer = weight_sum - correction * (weights**2).sum() / weight_sum
        expected = (centered * weights) @ centered.conj().T / normalizer

        actual = cov(
            xp.asarray(m),
            correction=correction,
            aweights=xp.asarray(weights),
        )
        assert_close(actual, xp.asarray(expected))

    def test_empty(self, xp: ArrayNamespace):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", RuntimeWarning)
            warnings.simplefilter("always", UserWarning)
            assert_equal(
                cov(xp.asarray([], dtype=xp.float64)),
                xp.asarray(xp.nan, dtype=xp.float64),
            )
            assert_equal(
                cov(xp.reshape(xp.asarray([], dtype=xp.float64), (0, 2))),
                xp.reshape(xp.asarray([], dtype=xp.float64), (0, 0)),
            )
            assert_equal(
                cov(xp.reshape(xp.asarray([], dtype=xp.float64), (2, 0))),
                xp.asarray([[xp.nan, xp.nan], [xp.nan, xp.nan]], dtype=xp.float64),
            )

    def test_combination(self, xp: ArrayNamespace):
        x = xp.asarray([-2.1, -1, 4.3], dtype=xp.float64)
        y = xp.asarray([3, 1.1, 0.12], dtype=xp.float64)
        X = xp.stack((x, y), axis=0)
        desired = xp.asarray([[11.71, -4.286], [-4.286, 2.144133]], dtype=xp.float64)
        assert_close(cov(X), desired, rtol=1e-6)
        assert_close(cov(x), xp.asarray(11.71, dtype=xp.float64))
        assert_close(cov(y), xp.asarray(2.144133, dtype=xp.float64), rtol=1e-6)

    @pytest.mark.xfail_xp_backend(
        Backend.TORCH, reason="torch.cov does not support tensors on meta device"
    )
    def test_device(self, xp: ArrayNamespace, device: Device):
        x = xp.asarray([1, 2, 3], device=device)
        assert get_device(cov(x)) == device

    @pytest.mark.skip_xp_backend(Backend.NUMPY_READONLY, reason="xp=xp")
    def test_xp(self, xp: ArrayNamespace):
        assert_close(
            cov(
                xp.asarray([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]], dtype=xp.float64).T,
                xp=xp,
            ),
            xp.asarray([[1.0, -1.0], [-1.0, 1.0]], dtype=xp.float64),
        )

    def test_batch(self, xp: ArrayNamespace):
        rng = np.random.default_rng(8847643423)
        batch_shape = (3, 4)
        n_var, n_obs = 3, 20
        m = rng.random((*batch_shape, n_var, n_obs))
        res = cov(xp.asarray(m))
        ref_list = [np.cov(m_) for m_ in np.reshape(m, (-1, n_var, n_obs))]
        ref = np.reshape(np.stack(ref_list), (*batch_shape, n_var, n_var))
        assert_close(res, xp.asarray(ref))

    @pytest.mark.parametrize("bias", [True, False, 0, 1])
    def test_bias(self, xp: ArrayNamespace, bias: bool):
        # `bias` maps to `correction`: bias=True -> correction=0, bias=False -> 1.
        x = np.array([-2.1, -1, 4.3])
        y = np.array([3, 1.1, 0.12])
        X = np.stack((x, y), axis=0)
        ref = np.cov(X, bias=bias)
        assert_close(
            cov(xp.asarray(X, dtype=xp.float64), correction=0 if bias else 1),
            xp.asarray(ref, dtype=xp.float64),
            rtol=1e-6,
        )

    @pytest.mark.parametrize("bias", [True, False, 0, 1])
    def test_bias_batch(self, xp: ArrayNamespace, bias: bool):
        rng = np.random.default_rng(8847643423)
        batch_shape = (3, 4)
        n_var, n_obs = 3, 20
        m = rng.random((*batch_shape, n_var, n_obs))
        res = cov(xp.asarray(m), correction=0 if bias else 1)
        ref_list = [np.cov(m_, bias=bias) for m_ in np.reshape(m, (-1, n_var, n_obs))]
        ref = np.reshape(np.stack(ref_list), (*batch_shape, n_var, n_var))
        assert_close(res, xp.asarray(ref))

    def test_correction(self, xp: ArrayNamespace):
        rng = np.random.default_rng(20260417)
        m = rng.random((3, 20))
        for correction in (0, 1, 2):
            ref = np.cov(m, ddof=correction)
            res = cov(xp.asarray(m), correction=correction)
            assert_close(res, xp.asarray(ref))

    def test_correction_float(self, xp: ArrayNamespace):
        # Float correction: reference computed by hand (numpy.cov rejects
        # non-integer ddof; our generic path supports it).
        rng = np.random.default_rng(20260417)
        m = rng.random((3, 20))
        n = m.shape[-1]
        centered = m - m.mean(axis=-1, keepdims=True)
        ref = centered @ centered.T / (n - 1.5)
        res = cov(xp.asarray(m), correction=1.5)
        assert_close(res, xp.asarray(ref))

    def test_axis(self, xp: ArrayNamespace):
        rng = np.random.default_rng(20260417)
        m = rng.random((20, 3))  # observations on axis 0
        ref = np.cov(m, rowvar=False)
        res = cov(xp.asarray(m), axis=0)
        assert_close(res, xp.asarray(ref))
        res_neg = cov(xp.asarray(m), axis=-2)
        assert_close(res_neg, xp.asarray(ref))

    def test_frequency_weights(self, xp: ArrayNamespace):
        rng = np.random.default_rng(20260417)
        m = rng.random((3, 10))
        fw = np.asarray([1, 2, 1, 3, 1, 2, 1, 1, 2, 1], dtype=np.int64)
        ref = np.cov(m, fweights=fw)
        res = cov(xp.asarray(m), fweights=xp.asarray(fw))
        assert_close(res, xp.asarray(ref))

    def test_weights(self, xp: ArrayNamespace):
        rng = np.random.default_rng(20260417)
        m = rng.random((3, 10))
        aw = rng.random(10)
        ref = np.cov(m, aweights=aw)
        res = cov(xp.asarray(m), aweights=xp.asarray(aw))
        assert_close(res, xp.asarray(ref))

    def test_both_weights(self, xp: ArrayNamespace):
        rng = np.random.default_rng(20260417)
        m = rng.random((3, 10))
        fw = np.asarray([1, 2, 1, 3, 1, 2, 1, 1, 2, 1], dtype=np.int64)
        aw = rng.random(10)
        for correction in (0, 1, 2):
            ref = np.cov(m, ddof=correction, fweights=fw, aweights=aw)
            res = cov(
                xp.asarray(m),
                correction=correction,
                fweights=xp.asarray(fw),
                aweights=xp.asarray(aw),
            )
            assert_close(res, xp.asarray(ref))

    def test_batch_with_weights(self, xp: ArrayNamespace):
        rng = np.random.default_rng(20260417)
        batch_shape = (2, 3)
        n_var, n_obs = 3, 15
        m = rng.random((*batch_shape, n_var, n_obs))
        aw = rng.random(n_obs)
        res = cov(xp.asarray(m), aweights=xp.asarray(aw))
        ref_list = [np.cov(m_, aweights=aw) for m_ in np.reshape(m, (-1, n_var, n_obs))]
        ref = np.reshape(np.stack(ref_list), (*batch_shape, n_var, n_var))
        assert_close(res, xp.asarray(ref))

    def test_axis_with_weights(self, xp: ArrayNamespace):
        # axis=-2 (observations on first of 2D) combined with weights:
        # verifies that moveaxis and weight alignment cooperate.
        rng = np.random.default_rng(20260417)
        m = rng.random((15, 3))  # observations on axis 0
        aw = rng.random(15)
        fw = np.asarray([1, 2, 1, 3, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1], dtype=np.int64)
        ref = np.cov(m, rowvar=False, fweights=fw, aweights=aw)
        res = cov(
            xp.asarray(m),
            axis=-2,
            fweights=xp.asarray(fw),
            aweights=xp.asarray(aw),
        )
        assert_close(res, xp.asarray(ref))

    def test_axis_out_of_bounds(self, xp: ArrayNamespace):
        m = xp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with pytest.raises(IndexError):
            _ = cov(m, axis=5)

    def test_weights_wrong_ndim(self, xp: ArrayNamespace):
        m = xp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        w2d = xp.asarray([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        # Non-integer correction forces the generic path where the
        # validation lives; native backends raise for the same reason.
        with pytest.raises((ValueError, TypeError)):
            _ = cov(m, correction=0.5, fweights=w2d)
        with pytest.raises((ValueError, TypeError)):
            _ = cov(m, correction=0.5, aweights=w2d)

    def test_weights_wrong_length(self, xp: ArrayNamespace):
        m = xp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        w_bad = xp.asarray([1.0, 1.0])  # expected length 3
        with pytest.raises((ValueError, RuntimeError)):
            _ = cov(m, correction=0.5, fweights=w_bad)
        with pytest.raises((ValueError, RuntimeError)):
            _ = cov(m, correction=0.5, aweights=w_bad)

    def test_weights_unknown_length(self, da: ArrayNamespace):
        m_np = np.asarray([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]])
        weights_np = np.asarray([1.0, 2.0, 3.0])
        keep_np = np.asarray([True, False, True])

        keep = da.asarray(keep_np)
        m = da.asarray(m_np)[:, keep]
        weights = da.asarray(weights_np)[keep]
        assert math.isnan(m.shape[-1])
        assert math.isnan(weights.shape[0])

        actual = cov(m, aweights=weights)
        desired = np.cov(m_np[:, keep_np], aweights=weights_np[keep_np])
        assert_close(actual, da.asarray(desired))

    def test_weights_dof_warning_eager(self):
        xp = array_namespace(cast(Array, cast(object, np.empty(0))))
        m = xp.asarray([[1.0, 2.0], [3.0, 4.0]])
        weights = xp.asarray([1.0, 1.0])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = cov(m, correction=2.5, aweights=weights)
        assert any(
            isinstance(warning.message, RuntimeWarning)
            and "Degrees of freedom <= 0" in str(warning.message)
            for warning in caught
        )

    def test_torch_autograd(self, torch: ArrayNamespace):
        # The batched (generic) path must not detach gradients or mutate the
        # input tensor in place, as `xp.asarray` does on torch.
        xp = torch
        rng = np.random.default_rng(20260417)
        m = xp.asarray(rng.random((4, 3, 20)), dtype=xp.float64)
        m.requires_grad_(True)
        m_before = m.detach().clone()
        # cov returns the array-api `Array` type; at runtime it is a torch
        # tensor, so cast to access autograd attributes without type errors.
        c = cast(Any, cov(m))  # batched -> generic path
        assert c.requires_grad
        assert m.requires_grad  # input tensor not mutated
        assert_equal(m.detach(), m_before)
        c.sum().backward()
        assert m.grad is not None
        assert bool(xp.all(xp.isfinite(m.grad)))


class TestNanMin:
    def test_simple(self, xp: ArrayNamespace):
        a = xp.asarray([[1, 2], [3, xp.nan]])

        # with the default `axis=None` a single scalar is returned
        res = nanmin(a)
        expected = 1.0
        assert res == expected

        res = nanmin(a, axis=0)
        expected = xp.asarray([1.0, 2.0])
        assert_equal(res, expected)

        res = nanmin(a, axis=1)
        expected = xp.asarray([1.0, 3.0])
        assert_equal(res, expected)

    def test_bigger(self, xp: ArrayNamespace):
        a = xp.asarray(
            [
                [1, xp.nan, 4, 5],
                [xp.nan, -2, xp.nan, -4],
                [2, 1, 3, xp.nan],
            ]
        )

        res = nanmin(a, axis=0)
        expected = xp.asarray([1.0, -2.0, 3.0, -4.0])
        assert_equal(res, expected)

        res = nanmin(a, axis=1)
        expected = xp.asarray([1.0, -4.0, 1.0])
        assert_equal(res, expected)

    def test_with_infinity(self, xp: ArrayNamespace):
        a = xp.asarray([0.1, 1.0, xp.nan, xp.inf])
        res = nanmin(a)
        expected = 0.1
        assert res == expected

        a = xp.asarray([0.1, 1.0, xp.nan, -xp.inf])
        res = nanmin(a)
        expected = -xp.inf
        assert res == expected

    def test_scalar(self, xp: ArrayNamespace):
        a = xp.asarray(1.0)
        assert nanmin(a) == 1.0

    @pytest.mark.filterwarnings("ignore:.*All-NaN slice*.:RuntimeWarning")
    def test_all_nan_slice_2d(self, xp: ArrayNamespace):
        a = xp.asarray(
            [
                [xp.nan, 5.0],
                [xp.nan, 2.0],
            ]
        )

        res = nanmin(a, axis=0, xp=xp)
        expected = xp.asarray([xp.nan, 2.0])
        assert_equal(res, expected)

    @pytest.mark.skip_xp_backend(
        Backend.TORCH, reason="torch.nanmin does not support tensors on meta device"
    )
    @pytest.mark.parametrize("axis", [None, 0, 1])
    def test_device(self, axis: int | None, xp: ArrayNamespace, device: Device):
        a = xp.asarray([[4, xp.nan, 1], [2, 5, xp.nan]], device=device)
        res = nanmin(a, axis=axis)
        assert get_device(res) == device

    @pytest.mark.parametrize(
        ("axis", "expected_list"), [(0, [2.0, 3.0, 1.0]), (1, [1.0, 2.0])]
    )
    def test_xp(self, axis: int | None, expected_list: list[float], xp: ArrayNamespace):
        a = xp.asarray([[4, xp.nan, 1], [2, 3, xp.nan]])
        res = nanmin(a, axis=axis, xp=xp)
        expected = xp.asarray(expected_list)
        assert_equal(res, expected)


class TestNanMax:
    def test_simple(self, xp: ArrayNamespace):
        a = xp.asarray([[5, 3], [6, xp.nan]])

        # with the default `axis=None` a single scalar is returned
        res = nanmax(a)
        expected = 6.0
        assert res == expected

        res = nanmax(a, axis=0)
        expected = xp.asarray([6.0, 3.0])
        assert_equal(res, expected)

        res = nanmax(a, axis=1)
        expected = xp.asarray([5.0, 6.0])
        assert_equal(res, expected)

    def test_bigger(self, xp: ArrayNamespace):
        a = xp.asarray(
            [
                [1, xp.nan, 4, 5],
                [xp.nan, 2, xp.nan, 4],
                [6, 1, 3, xp.nan],
            ]
        )

        res = nanmax(a, axis=0)
        expected = xp.asarray([6.0, 2.0, 4.0, 5.0])
        assert_equal(res, expected)

        res = nanmax(a, axis=1)
        expected = xp.asarray([5.0, 4.0, 6.0])
        assert_equal(res, expected)

    def test_with_infinity(self, xp: ArrayNamespace):
        a = xp.asarray([0.1, 5.0, xp.nan, -xp.inf])
        res = nanmax(a)
        expected = 5.0
        assert res == expected

        a = xp.asarray([3.0, 10.0, xp.nan, xp.inf])
        res = nanmax(a)
        expected = xp.inf
        assert res == expected

    def test_scalar(self, xp: ArrayNamespace):
        a = xp.asarray(1.0)
        assert nanmax(a) == 1.0

    @pytest.mark.filterwarnings("ignore:.*All-NaN slice*.:RuntimeWarning")
    def test_all_nan_slice_2d(self, xp: ArrayNamespace):
        a = xp.asarray(
            [
                [xp.nan, 5.0],
                [xp.nan, 2.0],
            ]
        )

        res = nanmax(a, axis=0, xp=xp)
        expected = xp.asarray([xp.nan, 5.0])
        assert_equal(res, expected)

    @pytest.mark.skip_xp_backend(
        Backend.TORCH, reason="torch.nanmax does not support tensors on meta device"
    )
    @pytest.mark.parametrize("axis", [None, 0, 1])
    def test_device(self, axis: int | None, xp: ArrayNamespace, device: Device):
        a = xp.asarray([[4, xp.nan, 1], [2, 5, xp.nan]], device=device)
        res = nanmax(a, axis=axis)
        assert get_device(res) == device

    @pytest.mark.parametrize(
        ("axis", "expected_list"), [(0, [4.0, 3.0, 1.0]), (1, [4.0, 3.0])]
    )
    def test_xp(self, axis: int | None, expected_list: list[float], xp: ArrayNamespace):
        a = xp.asarray([[4, xp.nan, 1], [2, 3, xp.nan]])
        res = nanmax(a, axis=axis, xp=xp)
        expected = xp.asarray(expected_list)
        assert_equal(res, expected)


class TestNanSum:
    def test_simple(self, xp: ArrayNamespace):
        a = xp.asarray([[1.0, 2.0], [3.0, xp.nan]])

        res = nansum(a)
        expected = 6.0
        assert res == expected

        res = nansum(a, axis=0)
        expected = xp.asarray([4.0, 2.0])
        assert_equal(res, expected)

        res = nansum(a, axis=1)
        expected = xp.asarray([3.0, 3.0])
        assert_equal(res, expected)

    def test_bigger(self, xp: ArrayNamespace):
        a = xp.asarray(
            [
                [1.0, xp.nan, 4.0, 5.0],
                [xp.nan, -2.0, xp.nan, -4.0],
                [2.0, 1.0, 3.0, xp.nan],
            ]
        )

        res = nansum(a, axis=0)
        expected = xp.asarray([3.0, -1.0, 7.0, 1.0])
        assert_equal(res, expected)

        res = nansum(a, axis=1)
        expected = xp.asarray([10.0, -6.0, 6.0])
        assert_equal(res, expected)

    def test_all_nan_slice(self, xp: ArrayNamespace):
        a = xp.asarray([[xp.nan, 1.0], [xp.nan, xp.nan]])

        res = nansum(a, axis=0)
        expected = xp.asarray([0.0, 1.0])
        assert_equal(res, expected)

    def test_scalar(self, xp: ArrayNamespace):
        a = xp.asarray(1.0)
        assert nansum(a) == 1.0

    @pytest.mark.skip_xp_backend(
        Backend.TORCH, reason="torch.nansum does not support tensors on meta device"
    )
    @pytest.mark.parametrize("axis", [None, 0, 1])
    def test_device(self, axis: int | None, xp: ArrayNamespace, device: Device):
        a = xp.asarray([[4.0, xp.nan, 1.0], [2.0, 5.0, xp.nan]], device=device)
        res = nansum(a, axis=axis)
        assert get_device(res) == device

    @pytest.mark.parametrize(
        ("axis", "expected_list"), [(0, [6.0, 3.0, 1.0]), (1, [5.0, 5.0])]
    )
    def test_xp(self, axis: int | None, expected_list: list[float], xp: ArrayNamespace):
        a = xp.asarray([[4.0, xp.nan, 1.0], [2.0, 3.0, xp.nan]])
        res = nansum(a, axis=axis, xp=xp)
        expected = xp.asarray(expected_list)
        assert_equal(res, expected)


class TestNanMean:
    def test_simple(self, xp: ArrayNamespace):
        a = xp.asarray([[1.0, 2.0], [3.0, xp.nan]])

        res = nanmean(a)
        assert res == 2.0

        res = nanmean(a, axis=0)
        expected = xp.asarray([2.0, 2.0])
        assert_equal(res, expected)

        res = nanmean(a, axis=1)
        expected = xp.asarray([1.5, 3.0])
        assert_equal(res, expected)

    def test_bigger(self, xp: ArrayNamespace):
        a = xp.asarray(
            [
                [1.0, xp.nan, 4.0, 5.0],
                [xp.nan, -2.0, xp.nan, -4.0],
                [2.0, 1.0, 3.0, xp.nan],
            ]
        )

        res = nanmean(a, axis=0)
        expected = xp.asarray([1.5, -0.5, 3.5, 0.5])
        assert_equal(res, expected)

        res = nanmean(a, axis=1)
        expected = xp.asarray([3.3333333, -3.0, 2.0])
        assert_close(res, expected)

    @pytest.mark.filterwarnings("ignore:.*Mean of empty slice.*:RuntimeWarning")
    def test_all_nan_slice(self, xp: ArrayNamespace):
        a = xp.asarray([[xp.nan, 1.0], [xp.nan, 3.0], [xp.nan, xp.nan]])

        res = nanmean(a, axis=0, xp=xp)
        expected = xp.asarray([xp.nan, 2.0])
        assert_equal(res, expected)

    def test_scalar(self, xp: ArrayNamespace):
        a = xp.asarray(1.0)
        assert nanmean(a) == 1.0

    @pytest.mark.skip_xp_backend(
        Backend.TORCH, reason="torch.nanmean does not support tensors on meta device"
    )
    @pytest.mark.parametrize("axis", [None, 0, 1])
    def test_device(self, axis: int | None, xp: ArrayNamespace, device: Device):
        a = xp.asarray([[4.0, xp.nan, 1.0], [2.0, 5.0, xp.nan]], device=device)
        res = nanmean(a, axis=axis)
        assert get_device(res) == device

    @pytest.mark.parametrize(
        ("axis", "expected_list"), [(0, [3.0, 5.0, 1.0]), (1, [2.5, 3.5])]
    )
    def test_xp(self, axis: int | None, expected_list: list[float], xp: ArrayNamespace):
        a = xp.asarray([[4.0, xp.nan, 1.0], [2.0, 5.0, xp.nan]])
        res = nanmean(a, axis=axis, xp=xp)
        expected = xp.asarray(expected_list)
        assert_equal(res, expected)
