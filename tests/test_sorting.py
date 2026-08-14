import numpy as np
import pytest
from typing_extensions import override

from array_api_extra import argpartition, partition
from array_api_extra._lib._backends import Backend
from array_api_extra._lib._typing import Array, ArrayNamespace
from array_api_extra.testing import lazy_xp_function

lazy_xp_function(argpartition)
lazy_xp_function(partition)


class TestPartition:
    @classmethod
    def _assert_valid_partition(
        cls,
        x_np: np.ndarray | None,
        k: int,
        y: Array,
        xp: ArrayNamespace,
        axis: int | None = -1,
    ):
        """
        x_np : input array
        k : int
        y : output array returned by the partition function to test
        """
        if x_np is not None:
            assert y.shape == np.partition(x_np, k, axis=axis).shape
        if y.ndim != 1 and axis == 0:
            assert isinstance(y.shape[1], int)
            for i in range(y.shape[1]):
                cls._assert_valid_partition(None, k, y[:, i, ...], xp, axis=0)
        elif y.ndim != 1:
            assert axis is not None
            axis = axis - 1 if axis != -1 else -1
            assert isinstance(y.shape[0], int)
            for i in range(y.shape[0]):
                cls._assert_valid_partition(None, k, y[i, ...], xp, axis=axis)
        else:
            if k > 0:
                assert xp.max(y[:k]) <= y[k]
            assert y[k] <= xp.min(y[k:])

    @classmethod
    def _partition(
        cls, x: np.ndarray, k: int, xp: ArrayNamespace, axis: int | None = -1
    ):
        return partition(xp.asarray(x), k, axis=axis)

    def _test_1d(self, xp: ArrayNamespace):
        rng = np.random.default_rng()
        for n in [2, 3, 4, 5, 7, 10, 20, 50, 100, 1_000]:
            k = int(rng.integers(n))
            x1 = rng.integers(n, size=n)
            y = self._partition(x1, k, xp)
            self._assert_valid_partition(x1, k, y, xp)
            x2 = rng.random(n)
            y = self._partition(x2, k, xp)
            self._assert_valid_partition(x2, k, y, xp)

    def _test_nd(self, xp: ArrayNamespace, ndim: int):
        rng = np.random.default_rng()

        for n in [2, 3, 5, 10, 20, 100]:
            base_shape = [int(v) for v in rng.integers(1, 4, size=ndim)]
            k = int(rng.integers(n))

            for i in range(ndim):
                shape = base_shape[:]
                shape[i] = n
                x = rng.integers(n, size=tuple(shape))
                y = self._partition(x, k, xp, axis=i)
                self._assert_valid_partition(x, k, y, xp, axis=i)

            z = rng.random(tuple(base_shape))
            k = int(rng.integers(z.size))
            y = self._partition(z, k, xp, axis=None)
            self._assert_valid_partition(z, k, y, xp, axis=None)

    def _test_input_validation(self, xp: ArrayNamespace):
        with pytest.raises(TypeError):
            _ = self._partition(np.asarray(1), 1, xp)
        with pytest.raises(ValueError, match="out of bounds"):
            _ = self._partition(np.asarray([1, 2]), 3, xp)

    def test_1d(self, xp: ArrayNamespace):
        self._test_1d(xp)

    @pytest.mark.parametrize("ndim", [2, 3, 4])
    def test_nd(self, xp: ArrayNamespace, ndim: int):
        self._test_nd(xp, ndim)

    def test_input_validation(self, xp: ArrayNamespace):
        self._test_input_validation(xp)


@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no argsort")
class TestArgpartition(TestPartition):
    @classmethod
    @override
    def _partition(
        cls, x: np.ndarray, k: int, xp: ArrayNamespace, axis: int | None = -1
    ):
        arr = xp.asarray(x)
        indices = argpartition(arr, k, axis=axis)
        if axis is None:
            arr = xp.reshape(arr, shape=(-1,))
            return arr[indices]
        if arr.ndim == 1:
            return arr[indices]
        return cls._take_along_axis(arr, indices, axis=axis, xp=xp)

    @classmethod
    def _take_along_axis(
        cls, arr: Array, indices: Array, axis: int, xp: ArrayNamespace
    ):
        if hasattr(xp, "take_along_axis"):
            return xp.take_along_axis(arr, indices, axis=axis)
        if arr.ndim == 1:
            return arr[indices]
        if axis == 0:
            assert isinstance(arr.shape[1], int)
            arrs = []
            for i in range(arr.shape[1]):
                arrs.append(
                    cls._take_along_axis(
                        arr[:, i, ...], indices[:, i, ...], axis=0, xp=xp
                    )
                )
            return xp.stack(arrs, axis=1)
        axis = axis - 1 if axis != -1 else -1
        assert isinstance(arr.shape[0], int)
        arrs = []
        for i in range(arr.shape[0]):
            arrs.append(
                cls._take_along_axis(arr[i, ...], indices[i, ...], axis=axis, xp=xp)
            )
        return xp.stack(arrs, axis=0)

    @override
    def test_1d(self, xp: ArrayNamespace):
        self._test_1d(xp)

    @pytest.mark.parametrize("ndim", [2, 3, 4])
    @override
    def test_nd(self, xp: ArrayNamespace, ndim: int):
        self._test_nd(xp, ndim)

    @override
    def test_input_validation(self, xp: ArrayNamespace):
        self._test_input_validation(xp)
