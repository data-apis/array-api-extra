import pytest

from array_api_extra import kron
from array_api_extra._lib._compat import device as get_device
from array_api_extra._lib._typing import ArrayNamespace, Device
from array_api_extra.testing import assert_equal, lazy_xp_function

lazy_xp_function(kron)


class TestKron:
    def test_basic(self, xp: ArrayNamespace):
        # Using 0-dimensional array
        a = xp.asarray(1)
        b = xp.asarray([[1, 2], [3, 4]])
        assert_equal(kron(a, b), b)
        assert_equal(kron(b, a), b)

        # Using 1-dimensional array
        a = xp.asarray([3])
        b = xp.asarray([[1, 2], [3, 4]])
        k = xp.asarray([[3, 6], [9, 12]])
        assert_equal(kron(a, b), k)
        assert_equal(kron(b, a), k)

        # Using 3-dimensional array
        a = xp.asarray([[[1]], [[2]]])
        b = xp.asarray([[1, 2], [3, 4]])
        k = xp.asarray([[[1, 2], [3, 4]], [[2, 4], [6, 8]]])
        assert_equal(kron(a, b), k)
        assert_equal(kron(b, a), k)

    def test_kron_smoke(self, xp: ArrayNamespace):
        a = xp.ones((3, 3))
        b = xp.ones((3, 3))
        k = xp.ones((9, 9))

        assert_equal(kron(a, b), k)

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
        self, xp: ArrayNamespace, shape_a: tuple[int, ...], shape_b: tuple[int, ...]
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

    def test_python_scalar(self, xp: ArrayNamespace):
        a = 1
        # Test no dtype promotion to xp.asarray(a); use b.dtype
        b = xp.asarray([[1, 2], [3, 4]], dtype=xp.int16)
        assert_equal(kron(a, b), b)
        assert_equal(kron(b, a), b)
        assert_equal(kron(1, 1, xp=xp), xp.asarray(1))

    def test_all_python_scalars(self):
        with pytest.raises(TypeError, match=r"array_namespace requires .* array input"):
            _ = kron(1, 1)

    def test_device(self, xp: ArrayNamespace, device: Device):
        x1 = xp.asarray([1, 2, 3], device=device)
        x2 = xp.asarray([4, 5], device=device)
        assert get_device(kron(x1, x2)) == device

    def test_xp(self, xp: ArrayNamespace):
        a = xp.ones((3, 3))
        b = xp.ones((3, 3))
        k = xp.ones((9, 9))
        assert_equal(kron(a, b, xp=xp), k)
