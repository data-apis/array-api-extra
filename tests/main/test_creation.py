import numpy as np
import pytest

from array_api_extra import at, create_diagonal, one_hot
from array_api_extra._lib._backends import Backend
from array_api_extra._lib._compat import device as get_device
from array_api_extra._lib._helpers import eager_shape, ndindex
from array_api_extra._lib._typing import ArrayNamespace, Device
from array_api_extra.testing import assert_equal, lazy_xp_function

lazy_xp_function(create_diagonal)
lazy_xp_function(one_hot)


@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no arange", strict=False)
class TestOneHot:
    @pytest.mark.parametrize("n_dim", range(4))
    @pytest.mark.parametrize("num_classes", [1, 3, 10])
    def test_dims_and_classes(self, xp: ArrayNamespace, n_dim: int, num_classes: int):
        shape = tuple(range(2, 2 + n_dim))
        rng = np.random.default_rng(2347823)
        np_x = rng.integers(num_classes, size=shape)
        x = xp.asarray(np_x)
        y = one_hot(x, num_classes)
        assert y.shape == (*x.shape, num_classes)
        for *i_list, j in ndindex(*shape, num_classes):
            i = tuple(i_list)
            assert float(y[(*i, j)]) == (int(x[i]) == j)

    def test_basic(self, xp: ArrayNamespace):
        actual = one_hot(xp.asarray([0, 1, 2]), 3)
        expected = xp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        assert_equal(actual, expected)

        actual = one_hot(xp.asarray([1, 2, 0]), 3)
        expected = xp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        assert_equal(actual, expected)

    def test_2d(self, xp: ArrayNamespace):
        actual = one_hot(xp.asarray([[2, 1, 0], [1, 0, 2]]), 3, axis=1)
        expected = xp.asarray(
            [
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ]
        )
        assert_equal(actual, expected)

    @pytest.mark.skip_xp_backend(
        Backend.ARRAY_API_STRICTEST, reason="backend doesn't support Boolean indexing"
    )
    def test_abstract_size(self, xp: ArrayNamespace):
        x = xp.arange(5)
        x = x[x > 2]
        actual = one_hot(x, 5)
        expected = xp.asarray([[0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
        assert_equal(actual, expected)

    @pytest.mark.skip_xp_backend(
        Backend.TORCH_GPU, reason="Puts Pytorch into a bad state."
    )
    def test_out_of_bound(self, xp: ArrayNamespace):
        # Undefined behavior.  Either return zero, or raise.
        try:
            actual = one_hot(xp.asarray([-1, 3]), 3)
        except IndexError:
            return
        expected = xp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        assert_equal(actual, expected)

    @pytest.mark.parametrize(
        "int_dtype",
        ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"],
    )
    def test_int_types(self, xp: ArrayNamespace, int_dtype: str):
        dtype = getattr(xp, int_dtype)
        x = xp.asarray([0, 1, 2], dtype=dtype)
        actual = one_hot(x, 3)
        expected = xp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        assert_equal(actual, expected)

    def test_custom_dtype(self, xp: ArrayNamespace):
        actual = one_hot(xp.asarray([0, 1, 2], dtype=xp.int32), 3, dtype=xp.bool)
        expected = xp.asarray(
            [[True, False, False], [False, True, False], [False, False, True]]
        )
        assert_equal(actual, expected)

    def test_axis(self, xp: ArrayNamespace):
        expected = xp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]).T
        actual = one_hot(xp.asarray([1, 2, 0]), 3, axis=0)
        assert_equal(actual, expected)

        actual = one_hot(xp.asarray([1, 2, 0]), 3, axis=-2)
        assert_equal(actual, expected)

    def test_non_integer(self, xp: ArrayNamespace):
        with pytest.raises(TypeError):
            _ = one_hot(xp.asarray([1.0]), 3)

    def test_device(self, xp: ArrayNamespace, device: Device):
        x = xp.asarray([0, 1, 2], device=device)
        y = one_hot(x, 3)
        assert get_device(y) == device


@pytest.mark.skip_xp_backend(
    Backend.SPARSE, reason="read-only backend without .at support"
)
class TestCreateDiagonal:
    def test_1d_from_numpy(self, xp: ArrayNamespace):
        # from np.diag tests
        vals = 100 * xp.arange(5, dtype=xp.float64)
        b = xp.zeros((5, 5), dtype=xp.float64)
        for k in range(5):
            b = at(b)[k, k].set(vals[k])
        assert_equal(create_diagonal(vals), b)
        b = xp.zeros((7, 7), dtype=xp.float64)
        c = xp.asarray(b, copy=True)
        for k in range(5):
            b = at(b)[k, k + 2].set(vals[k])
            c = at(c)[k + 2, k].set(vals[k])
        assert_equal(create_diagonal(vals, offset=2), b)
        assert_equal(create_diagonal(vals, offset=-2), c)

    @pytest.mark.parametrize("n", range(1, 10))
    @pytest.mark.parametrize("offset", range(1, 10))
    def test_1d_from_scipy(self, xp: ArrayNamespace, n: int, offset: int):
        # from scipy._lib tests
        rng = np.random.default_rng(2347823)
        one = xp.asarray(1.0)
        x = rng.random(n)
        A = create_diagonal(xp.asarray(x, dtype=one.dtype), offset=offset)
        B = xp.asarray(np.diag(x, offset), dtype=one.dtype)
        assert_equal(A, B)

    def test_0d_raises(self, xp: ArrayNamespace):
        with pytest.raises(ValueError, match="1-dimensional"):
            _ = create_diagonal(xp.asarray(1))

    @pytest.mark.parametrize(
        "shape",
        [
            (0,),
            (10,),
            (0, 1),
            (1, 0),
            (0, 0),
            (2, 3),
            (4, 2, 1),
            (1, 1, 7),
            (0, 0, 1),
            (3, 2, 4, 5),
        ],
    )
    def test_nd(self, xp: ArrayNamespace, shape: tuple[int, ...]):
        rng = np.random.default_rng(2347823)
        b = xp.asarray(
            rng.integers((1 << 64) - 1, size=shape, dtype=np.uint64), dtype=xp.uint64
        )
        c = create_diagonal(b)
        zero = xp.zeros((), dtype=xp.uint64)
        assert c.shape == (*b.shape, b.shape[-1])
        for i in ndindex(*eager_shape(c)):
            assert_equal(c[i], b[i[:-1]] if i[-2] == i[-1] else zero)

    def test_device(self, xp: ArrayNamespace, device: Device):
        x = xp.asarray([1, 2, 3], device=device)
        assert get_device(create_diagonal(x)) == device

    def test_xp(self, xp: ArrayNamespace):
        x = xp.asarray([1, 2])
        y = create_diagonal(x, xp=xp)
        assert_equal(y, xp.asarray([[1, 0], [0, 2]]))
