import math

import numpy as np
import pytest

from array_api_extra import atleast_nd, broadcast_shapes, expand_dims, pad
from array_api_extra._lib._backends import Backend
from array_api_extra._lib._compat import device as get_device
from array_api_extra._lib._typing import ArrayNamespace, Device
from array_api_extra.testing import assert_equal, lazy_xp_function

lazy_xp_function(atleast_nd)
lazy_xp_function(broadcast_shapes)
lazy_xp_function(expand_dims)
lazy_xp_function(pad)


class TestAtLeastND:
    def test_0D(self, xp: ArrayNamespace):
        x = xp.asarray(1.0)

        y = atleast_nd(x, ndim=0)
        assert_equal(y, x)

        y = atleast_nd(x, ndim=1)
        assert_equal(y, xp.ones((1,)))

        y = atleast_nd(x, ndim=5)
        assert_equal(y, xp.ones((1, 1, 1, 1, 1)))

    @pytest.mark.parametrize(
        ("input_shape", "ndim", "expected_shape"),
        [
            ((1,), 0, (1,)),
            ((5,), 1, (5,)),
            ((2,), 2, (1, 2)),
            ((3,), 3, (1, 1, 3)),
            ((2,), 5, (1, 1, 1, 1, 2)),
        ],
    )
    def test_1D_shapes(
        self,
        input_shape: tuple[int],
        ndim: int,
        expected_shape: tuple[int],
        xp: ArrayNamespace,
    ):
        n = math.prod(input_shape)
        x = xp.asarray(np.arange(n).reshape(input_shape))
        y = atleast_nd(x, ndim=ndim)

        assert y.shape == expected_shape
        assert xp.sum(y) == int(n * (n - 1) / 2)

    def test_1D_values(self, xp: ArrayNamespace):
        x = xp.asarray([0, 1])

        y = atleast_nd(x, ndim=0)
        assert_equal(y, x)

        y = atleast_nd(x, ndim=1)
        assert_equal(y, x)

        y = atleast_nd(x, ndim=2)
        assert_equal(y, xp.asarray([[0, 1]]))

        y = atleast_nd(x, ndim=5)
        assert_equal(y, xp.asarray([[[[[0, 1]]]]]))

    @pytest.mark.parametrize(
        ("input_shape", "ndim", "expected_shape"),
        [
            ((2, 1), 0, (2, 1)),
            ((5, 2), 1, (5, 2)),
            ((2, 1), 2, (2, 1)),
            ((3, 1), 3, (1, 3, 1)),
            ((2, 8), 5, (1, 1, 1, 2, 8)),
        ],
    )
    def test_2D_shapes(
        self,
        input_shape: tuple[int],
        ndim: int,
        expected_shape: tuple[int],
        xp: ArrayNamespace,
    ):
        n = math.prod(input_shape)
        x = xp.asarray(np.arange(n).reshape(input_shape))
        y = atleast_nd(x, ndim=ndim)

        assert y.shape == expected_shape
        assert xp.sum(y) == int(n * (n - 1) / 2)

    def test_2D_values(self, xp: ArrayNamespace):
        x = xp.asarray([[3.0], [4.0]])

        y = atleast_nd(x, ndim=0)
        assert_equal(y, x)

        y = atleast_nd(x, ndim=2)
        assert_equal(y, x)

        y = atleast_nd(x, ndim=3)
        assert_equal(y, xp.asarray([[[3.0], [4.0]]]))

        y = atleast_nd(x, ndim=5)
        assert_equal(y, xp.asarray([[[[[3.0], [4.0]]]]]))

    @pytest.mark.parametrize(
        ("input_shape", "ndim", "expected_shape"),
        [
            ((2, 1, 1), 0, (2, 1, 1)),
            ((1, 5, 2), 1, (1, 5, 2)),
            ((2, 1, 1), 2, (2, 1, 1)),
            ((1, 3, 1), 3, (1, 3, 1)),
            ((2, 8, 1), 5, (1, 1, 2, 8, 1)),
        ],
    )
    def test_3D_shapes(
        self,
        input_shape: tuple[int],
        ndim: int,
        expected_shape: tuple[int],
        xp: ArrayNamespace,
    ):
        n = math.prod(input_shape)
        x = xp.asarray(np.arange(n).reshape(input_shape))
        y = atleast_nd(x, ndim=ndim)

        assert y.shape == expected_shape
        assert xp.sum(y) == int(n * (n - 1) / 2)

    def test_3D_values(self, xp: ArrayNamespace):
        x = xp.asarray([[[3.0], [2.0]]])

        y = atleast_nd(x, ndim=0)
        assert_equal(y, x)

        y = atleast_nd(x, ndim=2)
        assert_equal(y, x)

        y = atleast_nd(x, ndim=3)
        assert_equal(y, x)

        y = atleast_nd(x, ndim=5)
        assert_equal(y, xp.asarray([[[[[3.0], [2.0]]]]]))

    @pytest.mark.parametrize(
        ("input_shape", "ndim", "expected_shape"),
        [
            ((2, 1, 1, 2, 1), 0, (2, 1, 1, 2, 1)),
            ((1, 5, 2, 3, 2), 2, (1, 5, 2, 3, 2)),
            ((2, 1, 1, 5, 2), 5, (2, 1, 1, 5, 2)),
            ((1, 3, 1, 2, 1), 6, (1, 1, 3, 1, 2, 1)),
            ((2, 8, 1, 9, 8), 9, (1, 1, 1, 1, 2, 8, 1, 9, 8)),
        ],
    )
    def test_5D_shapes(
        self,
        input_shape: tuple[int],
        ndim: int,
        expected_shape: tuple[int],
        xp: ArrayNamespace,
    ):
        n = math.prod(input_shape)
        x = xp.asarray(np.arange(n).reshape(input_shape))
        y = atleast_nd(x, ndim=ndim)

        assert y.shape == expected_shape
        assert xp.sum(y) == int(n * (n - 1) / 2)

    def test_5D_values(self, xp: ArrayNamespace):
        x = xp.asarray([[[[[3.0]], [[2.0]]]]])

        y = atleast_nd(x, ndim=0)
        assert_equal(y, x)

        y = atleast_nd(x, ndim=4)
        assert_equal(y, x)

        y = atleast_nd(x, ndim=5)
        assert_equal(y, x)

        y = atleast_nd(x, ndim=6)
        assert_equal(y, xp.asarray([[[[[[3.0]], [[2.0]]]]]]))

        y = atleast_nd(x, ndim=9)
        assert_equal(y, xp.asarray([[[[[[[[[3.0]], [[2.0]]]]]]]]]))


@pytest.mark.filterwarnings("ignore:.*removed in v1.0.0.*:DeprecationWarning")
class TestBroadcastShapes:
    def test_delegates_known_integer_shapes(self, monkeypatch: pytest.MonkeyPatch):
        calls = []

        def mock_broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]:
            calls.append(shapes)
            return (99,)

        monkeypatch.setattr(np, "broadcast_shapes", mock_broadcast_shapes)

        assert broadcast_shapes((2,), (1,), xp=np) == (99,)
        assert calls == [((2,), (1,))]

    def test_fallback_without_xp(self, monkeypatch: pytest.MonkeyPatch):
        def mock_broadcast_shapes(*_shapes: tuple[int, ...]) -> tuple[int, ...]:
            msg = "Native delegation should not be used without xp"
            raise AssertionError(msg)

        monkeypatch.setattr(np, "broadcast_shapes", mock_broadcast_shapes)

        assert broadcast_shapes((2,), (1,)) == (2,)

    @pytest.mark.skip_xp_backend(Backend.NUMPY_READONLY, reason="xp=xp")
    def test_xp(self, xp: ArrayNamespace):
        assert broadcast_shapes((2, 3), (2, 1), xp=xp) == (2, 3)

    @pytest.mark.parametrize(
        "args",
        [
            (),
            ((),),
            ((), ()),
            ((1,),),
            ((1,), (1,)),
            ((2,), (1,)),
            ((3, 1, 4), (2, 1)),
            ((1, 1, 4), (2, 1)),
            ((1,), ()),
            ((), (2,), ()),
            ((0,),),
            ((0,), (1,)),
            ((2, 0), (1, 1)),
            ((2, 0, 3), (2, 1, 1)),
        ],
    )
    def test_simple(self, args: tuple[tuple[int, ...], ...]):
        expect = np.broadcast_shapes(*args)
        actual = broadcast_shapes(*args)
        assert actual == expect

    @pytest.mark.parametrize(
        "args",
        [
            ((2,), (3,)),
            ((2, 3), (1, 2)),
            ((2,), (0,)),
            ((2, 0, 2), (1, 3, 1)),
        ],
    )
    def test_fail(self, args: tuple[tuple[int, ...], ...]):
        match = "cannot be broadcast to a single shape"
        with pytest.raises(ValueError, match=match):
            _ = np.broadcast_shapes(*args)
        with pytest.raises(ValueError, match=match):
            _ = broadcast_shapes(*args)

    @pytest.mark.parametrize(
        "args",
        [
            ((None,), (None,)),
            ((math.nan,), (None,)),
            ((1, None, 2, 4), (2, 3, None, 1), (2, None, None, 4)),
            ((1, math.nan, 2), (4, 2, 3, math.nan), (4, 2, None, None)),
            ((math.nan, 1), (None, 2), (None, 2)),
        ],
    )
    def test_none(self, args: tuple[tuple[float | None, ...], ...]):
        expect = args[-1]
        actual = broadcast_shapes(*args[:-1])
        assert actual == expect


@pytest.mark.filterwarnings(r"ignore:.*removed in v1.0.0.*:DeprecationWarning")
class TestExpandDims:
    def test_single_axis(self, xp: ArrayNamespace):
        """Trivial case where xpx.expand_dims doesn't add anything to xp.expand_dims"""
        a = xp.asarray(np.reshape(np.arange(2 * 3 * 4 * 5), (2, 3, 4, 5)))
        for axis in range(-5, 4):
            b = expand_dims(a, axis=axis)
            assert_equal(b, xp.expand_dims(a, axis=axis))

    def test_axis_tuple(self, xp: ArrayNamespace):
        a = xp.empty((3, 3, 3))
        assert expand_dims(a, axis=(0, 1, 2)).shape == (1, 1, 1, 3, 3, 3)
        assert expand_dims(a, axis=(0, -1, -2)).shape == (1, 3, 3, 3, 1, 1)
        assert expand_dims(a, axis=(0, 3, 5)).shape == (1, 3, 3, 1, 3, 1)
        assert expand_dims(a, axis=(0, -3, -5)).shape == (1, 1, 3, 1, 3, 3)

    def test_axis_out_of_range(self, xp: ArrayNamespace):
        a = xp.empty((2, 3, 4, 5))
        with pytest.raises(IndexError, match="out of bounds"):
            _ = expand_dims(a, axis=-6)
        with pytest.raises(IndexError, match="out of bounds"):
            _ = expand_dims(a, axis=5)

        a = xp.empty((3, 3, 3))
        with pytest.raises(IndexError, match="out of bounds"):
            _ = expand_dims(a, axis=(0, -6))
        with pytest.raises(IndexError, match="out of bounds"):
            _ = expand_dims(a, axis=(0, 5))

    def test_repeated_axis(self, xp: ArrayNamespace):
        a = xp.empty((3, 3, 3))
        with pytest.raises(ValueError, match="Duplicate dimensions"):
            _ = expand_dims(a, axis=(1, 1))

    def test_positive_negative_repeated(self, xp: ArrayNamespace):
        # https://github.com/data-apis/array-api/issues/760#issuecomment-1989449817
        a = xp.empty((2, 3, 4, 5))
        with pytest.raises(ValueError, match="Duplicate dimensions"):
            _ = expand_dims(a, axis=(3, -3))

    def test_device(self, xp: ArrayNamespace, device: Device):
        x = xp.asarray([1, 2, 3], device=device)
        assert get_device(expand_dims(x, axis=0)) == device

    def test_xp(self, xp: ArrayNamespace):
        x = xp.asarray([1, 2, 3])
        y = expand_dims(x, axis=(0, 1, 2), xp=xp)
        assert y.shape == (1, 1, 1, 3)


class TestPad:
    def test_simple(self, xp: ArrayNamespace):
        a = xp.asarray([1, 2, 3])
        padded = pad(a, 2)
        assert_equal(padded, xp.asarray([0, 0, 1, 2, 3, 0, 0]))

    @pytest.mark.xfail_xp_backend(
        Backend.SPARSE, reason="constant_values can only be equal to fill value"
    )
    def test_fill_value(self, xp: ArrayNamespace):
        a = xp.asarray([1, 2, 3])
        padded = pad(a, 2, constant_values=42)
        assert_equal(padded, xp.asarray([42, 42, 1, 2, 3, 42, 42]))

    def test_ndim(self, xp: ArrayNamespace):
        a = xp.asarray(np.reshape(np.arange(2 * 3 * 4), (2, 3, 4)))
        padded = pad(a, 2)
        assert padded.shape == (6, 7, 8)

    def test_mode_not_implemented(self, xp: ArrayNamespace):
        a = xp.asarray([1, 2, 3])
        with pytest.raises(NotImplementedError, match="Only `'constant'`"):
            _ = pad(a, 2, mode="edge")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

    def test_device(self, xp: ArrayNamespace, device: Device):
        a = xp.asarray(0.0, device=device)
        assert get_device(pad(a, 2)) == device

    def test_xp(self, xp: ArrayNamespace):
        padded = pad(xp.asarray(0), 1, xp=xp)
        assert_equal(padded, xp.asarray(0))

    def test_tuple_width(self, xp: ArrayNamespace):
        a = xp.asarray(np.reshape(np.arange(12), (3, 4)))
        padded = pad(a, (1, 0))
        assert padded.shape == (4, 5)

        padded = pad(a, (1, 2))
        assert padded.shape == (6, 7)

        with pytest.raises((ValueError, RuntimeError)):
            _ = pad(a, [(1, 2, 3)])  # type: ignore[list-item]  # pyright: ignore[reportArgumentType]

    def test_sequence_of_tuples_width(self, xp: ArrayNamespace):
        a = xp.asarray(np.reshape(np.arange(12), (3, 4)))

        padded = pad(a, ((1, 0), (0, 2)))
        assert padded.shape == (4, 6)
        padded = pad(a, ((1, 0), (0, 0)))
        assert padded.shape == (4, 4)
