import math
import warnings
from types import ModuleType
from typing import Any, Literal, cast, get_args

import hypothesis
import hypothesis.extra.numpy as npst
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from typing_extensions import override

from array_api_extra import (
    apply_where,
    argpartition,
    at,
    atleast_nd,
    broadcast_shapes,
    cov,
    create_diagonal,
    default_dtype,
    expand_dims,
    isclose,
    isin,
    kron,
    nan_to_num,
    nunique,
    one_hot,
    pad,
    partition,
    quantile,
    setdiff1d,
    sinc,
)
from array_api_extra._lib._backends import NUMPY_VERSION, Backend
from array_api_extra._lib._testing import xp_assert_close, xp_assert_equal
from array_api_extra._lib._utils._compat import (
    device as get_device,
)
from array_api_extra._lib._utils._helpers import eager_shape, ndindex
from array_api_extra._lib._utils._typing import Array, Device
from array_api_extra.testing import lazy_xp_function

lazy_xp_function(apply_where)
lazy_xp_function(atleast_nd)
lazy_xp_function(cov)
lazy_xp_function(create_diagonal)
lazy_xp_function(expand_dims)
lazy_xp_function(kron)
lazy_xp_function(nan_to_num)
lazy_xp_function(nunique)
lazy_xp_function(one_hot)
lazy_xp_function(pad)
# FIXME calls in1d which calls xp.unique_values without size
lazy_xp_function(setdiff1d, jax_jit=False)
lazy_xp_function(sinc)


class TestApplyWhere:
    @staticmethod
    def f1(x: Array, y: Array | int = 10) -> Array:
        return x + y

    @staticmethod
    def f2(x: Array, y: Array | int = 10) -> Array:
        return x - y

    def test_f1_f2(self, xp: ModuleType):
        x = xp.asarray([1, 2, 3, 4])
        cond = x % 2 == 0
        actual = apply_where(cond, x, self.f1, self.f2)
        expect = xp.where(cond, self.f1(x), self.f2(x))
        xp_assert_equal(actual, expect)

    def test_fill_value(self, xp: ModuleType):
        x = xp.asarray([1, 2, 3, 4])
        cond = x % 2 == 0
        actual = apply_where(x % 2 == 0, x, self.f1, fill_value=0)
        expect = xp.where(cond, self.f1(x), xp.asarray(0))
        xp_assert_equal(actual, expect)

        actual = apply_where(x % 2 == 0, x, self.f1, fill_value=xp.asarray(0))
        xp_assert_equal(actual, expect)

    def test_args_tuple(self, xp: ModuleType):
        x = xp.asarray([1, 2, 3, 4])
        y = xp.asarray([10, 20, 30, 40])
        cond = x % 2 == 0
        actual = apply_where(cond, (x, y), self.f1, self.f2)
        expect = xp.where(cond, self.f1(x, y), self.f2(x, y))
        xp_assert_equal(actual, expect)

    def test_broadcast(self, xp: ModuleType):
        x = xp.asarray([1, 2])
        y = xp.asarray([[10], [20], [30]])
        cond = xp.broadcast_to(xp.asarray(True), (4, 1, 1))

        actual = apply_where(cond, (x, y), self.f1, self.f2)
        expect = xp.where(cond, self.f1(x, y), self.f2(x, y))
        xp_assert_equal(actual, expect)

        actual = apply_where(
            cond,
            (x, y),
            lambda x, _: x,
            lambda _, y: y,
        )
        expect = xp.where(cond, x, y)
        xp_assert_equal(actual, expect)

        # Shaped fill_value
        actual = apply_where(cond, x, self.f1, fill_value=y)
        expect = xp.where(cond, self.f1(x), y)
        xp_assert_equal(actual, expect)

    def test_dtype_propagation(self, xp: ModuleType, library: Backend):
        x = xp.asarray([1, 2], dtype=xp.int8)
        y = xp.asarray([3, 4], dtype=xp.int16)
        cond = x % 2 == 0

        mxp = np if library is Backend.DASK else xp
        actual = apply_where(
            cond,
            (x, y),
            self.f1,
            lambda x, y: mxp.astype(x - y, xp.int64),  # pyright: ignore[reportArgumentType]
        )
        assert actual.dtype == xp.int64

        actual = apply_where(cond, y, self.f1, fill_value=5)
        assert actual.dtype == xp.int16

    @pytest.mark.parametrize("fill_value_raw", [3, [3, 4]])
    @pytest.mark.parametrize(
        ("fill_value_dtype", "expect_dtype"), [("int32", "int32"), ("int8", "int16")]
    )
    def test_dtype_propagation_fill_value(
        self,
        xp: ModuleType,
        fill_value_raw: int | list[int],
        fill_value_dtype: str,
        expect_dtype: str,
    ):
        x = xp.asarray([1, 2], dtype=xp.int16)
        cond = x % 2 == 0
        fill_value = xp.asarray(fill_value_raw, dtype=getattr(xp, fill_value_dtype))

        actual = apply_where(cond, x, self.f1, fill_value=fill_value)
        assert actual.dtype == getattr(xp, expect_dtype)

    def test_dont_overwrite_fill_value(self, xp: ModuleType):
        x = xp.asarray([1, 2])
        fill_value = xp.asarray([100, 200])
        actual = apply_where(x % 2 == 0, x, self.f1, fill_value=fill_value)
        xp_assert_equal(actual, xp.asarray([100, 12]))
        xp_assert_equal(fill_value, xp.asarray([100, 200]))

    @pytest.mark.skip_xp_backend(
        Backend.ARRAY_API_STRICTEST,
        reason="no boolean indexing -> run everywhere",
    )
    @pytest.mark.skip_xp_backend(
        Backend.SPARSE,
        reason="no indexing by sparse array -> run everywhere",
    )
    def test_dont_run_on_false(self, xp: ModuleType):
        x = xp.asarray([1.0, 2.0, 0.0])
        y = xp.asarray([0.0, 3.0, 4.0])
        # On NumPy, division by zero will trigger warnings
        actual = apply_where(
            x == 0,
            (x, y),
            lambda x, y: x / y,
            lambda x, y: y / x,
        )
        xp_assert_equal(actual, xp.asarray([0.0, 1.5, 0.0]))

    def test_bad_args(self, xp: ModuleType):
        x = xp.asarray([1, 2, 3, 4])
        cond = x % 2 == 0
        # Neither f2 nor fill_value
        with pytest.raises(TypeError, match="Exactly one of"):
            apply_where(cond, x, self.f1)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue]
        # Both f2 and fill_value
        with pytest.raises(TypeError, match="Exactly one of"):
            apply_where(cond, x, self.f1, self.f2, fill_value=0)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue]

    @pytest.mark.skip_xp_backend(Backend.NUMPY_READONLY, reason="xp=xp")
    def test_xp(self, xp: ModuleType):
        x = xp.asarray([1, 2, 3, 4])
        cond = x % 2 == 0
        actual = apply_where(cond, x, self.f1, self.f2, xp=xp)
        expect = xp.where(cond, self.f1(x), self.f2(x))
        xp_assert_equal(actual, expect)

    def test_device(self, xp: ModuleType, device: Device):
        x = xp.asarray([1, 2, 3, 4], device=device)
        y = apply_where(x % 2 == 0, x, self.f1, self.f2)
        assert get_device(y) == device
        y = apply_where(x % 2 == 0, x, self.f1, fill_value=0)
        assert get_device(y) == device
        y = apply_where(x % 2 == 0, x, self.f1, fill_value=x)
        assert get_device(y) == device

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")  # overflows, etc.
    @hypothesis.settings(
        # The xp and library fixtures are not regenerated between hypothesis iterations
        suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
        # JAX can take a long time to initialize on the first call
        deadline=None,
    )
    @given(
        n_arrays=st.integers(min_value=1, max_value=3),
        rng_seed=st.integers(min_value=1000000000, max_value=9999999999),
        dtype=npst.floating_dtypes(sizes=(32, 64)),
        p=st.floats(min_value=0, max_value=1),
        data=st.data(),
    )
    def test_hypothesis(
        self,
        n_arrays: int,
        rng_seed: int,
        dtype: np.dtype[Any],
        p: float,
        data: st.DataObject,
        xp: ModuleType,
        library: Backend,
    ):
        if (
            library.like(Backend.NUMPY)
            and NUMPY_VERSION < (2, 0)
            and dtype.type is np.float32
        ):
            pytest.xfail(reason="NumPy 1.x dtype promotion for scalars")

        mbs = npst.mutually_broadcastable_shapes(num_shapes=n_arrays + 1, min_side=0)
        input_shapes, _ = data.draw(mbs)
        cond_shape, *shapes = input_shapes

        # cupy/cupy#8382
        # https://github.com/jax-ml/jax/issues/26658
        elements = {"allow_subnormal": not library.like(Backend.CUPY, Backend.JAX)}

        fill_value = xp.asarray(
            data.draw(npst.arrays(dtype=dtype.type, shape=(), elements=elements))
        )
        float_fill_value = float(fill_value)
        if library is Backend.CUPY and dtype.type is np.float32:
            # Avoid data-dependent dtype promotion when encountering subnormals
            # close to the max float32 value
            float_fill_value = float(np.clip(float_fill_value, -1e38, 1e38))

        arrays = tuple(
            xp.asarray(
                data.draw(npst.arrays(dtype=dtype.type, shape=shape, elements=elements))
            )
            for shape in shapes
        )

        def f1(*args: Array) -> Array:
            return cast(Array, sum(args))

        def f2(*args: Array) -> Array:
            return cast(Array, sum(args) / 2)

        rng = np.random.default_rng(rng_seed)
        cond = xp.asarray(rng.random(size=cond_shape) > p)

        res1 = apply_where(cond, arrays, f1, fill_value=fill_value)
        res2 = apply_where(cond, arrays, f1, f2)
        res3 = apply_where(cond, arrays, f1, fill_value=float_fill_value)

        ref1 = xp.where(cond, f1(*arrays), fill_value)
        ref2 = xp.where(cond, f1(*arrays), f2(*arrays))
        ref3 = xp.where(cond, f1(*arrays), float_fill_value)

        xp_assert_close(res1, ref1, rtol=2e-16)
        xp_assert_equal(res2, ref2)
        xp_assert_equal(res3, ref3)


class TestAtLeastND:
    def test_0D(self, xp: ModuleType):
        x = xp.asarray(1.0)

        y = atleast_nd(x, ndim=0)
        xp_assert_equal(y, x)

        y = atleast_nd(x, ndim=1)
        xp_assert_equal(y, xp.ones((1,)))

        y = atleast_nd(x, ndim=5)
        xp_assert_equal(y, xp.ones((1, 1, 1, 1, 1)))

    def test_1D(self, xp: ModuleType):
        x = xp.asarray([0, 1])

        y = atleast_nd(x, ndim=0)
        xp_assert_equal(y, x)

        y = atleast_nd(x, ndim=1)
        xp_assert_equal(y, x)

        y = atleast_nd(x, ndim=2)
        xp_assert_equal(y, xp.asarray([[0, 1]]))

        y = atleast_nd(x, ndim=5)
        xp_assert_equal(y, xp.asarray([[[[[0, 1]]]]]))

    def test_2D(self, xp: ModuleType):
        x = xp.asarray([[3.0]])

        y = atleast_nd(x, ndim=0)
        xp_assert_equal(y, x)

        y = atleast_nd(x, ndim=2)
        xp_assert_equal(y, x)

        y = atleast_nd(x, ndim=3)
        xp_assert_equal(y, 3 * xp.ones((1, 1, 1)))

        y = atleast_nd(x, ndim=5)
        xp_assert_equal(y, 3 * xp.ones((1, 1, 1, 1, 1)))

    def test_3D(self, xp: ModuleType):
        x = xp.asarray([[[3.0], [2.0]]])

        y = atleast_nd(x, ndim=0)
        xp_assert_equal(y, x)

        y = atleast_nd(x, ndim=2)
        xp_assert_equal(y, x)

        y = atleast_nd(x, ndim=3)
        xp_assert_equal(y, x)

        y = atleast_nd(x, ndim=5)
        xp_assert_equal(y, xp.asarray([[[[[3.0], [2.0]]]]]))

    def test_5D(self, xp: ModuleType):
        x = xp.ones((1, 1, 1, 1, 1))

        y = atleast_nd(x, ndim=0)
        xp_assert_equal(y, x)

        y = atleast_nd(x, ndim=4)
        xp_assert_equal(y, x)

        y = atleast_nd(x, ndim=5)
        xp_assert_equal(y, x)

        y = atleast_nd(x, ndim=6)
        xp_assert_equal(y, xp.ones((1, 1, 1, 1, 1, 1)))

        y = atleast_nd(x, ndim=9)
        xp_assert_equal(y, xp.ones((1, 1, 1, 1, 1, 1, 1, 1, 1)))

    def test_device(self, xp: ModuleType, device: Device):
        x = xp.asarray([1, 2, 3], device=device)
        assert get_device(atleast_nd(x, ndim=2)) == device

    def test_xp(self, xp: ModuleType):
        x = xp.asarray(1.0)
        y = atleast_nd(x, ndim=1, xp=xp)
        xp_assert_equal(y, xp.ones((1,)))


class TestBroadcastShapes:
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


class TestCov:
    def test_basic(self, xp: ModuleType):
        xp_assert_close(
            cov(xp.asarray([[0, 2], [1, 1], [2, 0]], dtype=xp.float64).T),
            xp.asarray([[1.0, -1.0], [-1.0, 1.0]], dtype=xp.float64),
        )

    def test_complex(self, xp: ModuleType):
        actual = cov(xp.asarray([[1, 2, 3], [1j, 2j, 3j]], dtype=xp.complex128))
        expect = xp.asarray([[1.0, -1.0j], [1.0j, 1.0]], dtype=xp.complex128)
        xp_assert_close(actual, expect)

    @pytest.mark.xfail_xp_backend(Backend.JAX, reason="jax#32296")
    @pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="sparse#877")
    def test_empty(self, xp: ModuleType):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", RuntimeWarning)
            warnings.simplefilter("always", UserWarning)
            xp_assert_equal(
                cov(xp.asarray([], dtype=xp.float64)),
                xp.asarray(xp.nan, dtype=xp.float64),
            )
            xp_assert_equal(
                cov(xp.reshape(xp.asarray([], dtype=xp.float64), (0, 2))),
                xp.reshape(xp.asarray([], dtype=xp.float64), (0, 0)),
            )
            xp_assert_equal(
                cov(xp.reshape(xp.asarray([], dtype=xp.float64), (2, 0))),
                xp.asarray([[xp.nan, xp.nan], [xp.nan, xp.nan]], dtype=xp.float64),
            )

    def test_combination(self, xp: ModuleType):
        x = xp.asarray([-2.1, -1, 4.3], dtype=xp.float64)
        y = xp.asarray([3, 1.1, 0.12], dtype=xp.float64)
        X = xp.stack((x, y), axis=0)
        desired = xp.asarray([[11.71, -4.286], [-4.286, 2.144133]], dtype=xp.float64)
        xp_assert_close(cov(X), desired, rtol=1e-6)
        xp_assert_close(cov(x), xp.asarray(11.71, dtype=xp.float64))
        xp_assert_close(cov(y), xp.asarray(2.144133, dtype=xp.float64), rtol=1e-6)

    @pytest.mark.xfail_xp_backend(Backend.TORCH, reason="array-api-extra#455")
    def test_device(self, xp: ModuleType, device: Device):
        x = xp.asarray([1, 2, 3], device=device)
        assert get_device(cov(x)) == device

    @pytest.mark.skip_xp_backend(Backend.NUMPY_READONLY, reason="xp=xp")
    def test_xp(self, xp: ModuleType):
        xp_assert_close(
            cov(
                xp.asarray([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]], dtype=xp.float64).T,
                xp=xp,
            ),
            xp.asarray([[1.0, -1.0], [-1.0, 1.0]], dtype=xp.float64),
        )


@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no arange", strict=False)
class TestOneHot:
    @pytest.mark.parametrize("n_dim", range(4))
    @pytest.mark.parametrize("num_classes", [1, 3, 10])
    def test_dims_and_classes(self, xp: ModuleType, n_dim: int, num_classes: int):
        shape = tuple(range(2, 2 + n_dim))
        rng = np.random.default_rng(2347823)
        np_x = rng.integers(num_classes, size=shape)
        x = xp.asarray(np_x)
        y = one_hot(x, num_classes)
        assert y.shape == (*x.shape, num_classes)
        for *i_list, j in ndindex(*shape, num_classes):
            i = tuple(i_list)
            assert float(y[(*i, j)]) == (int(x[i]) == j)

    def test_basic(self, xp: ModuleType):
        actual = one_hot(xp.asarray([0, 1, 2]), 3)
        expected = xp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        xp_assert_equal(actual, expected)

        actual = one_hot(xp.asarray([1, 2, 0]), 3)
        expected = xp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        xp_assert_equal(actual, expected)

    def test_2d(self, xp: ModuleType):
        actual = one_hot(xp.asarray([[2, 1, 0], [1, 0, 2]]), 3, axis=1)
        expected = xp.asarray(
            [
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ]
        )
        xp_assert_equal(actual, expected)

    @pytest.mark.skip_xp_backend(
        Backend.ARRAY_API_STRICTEST, reason="backend doesn't support Boolean indexing"
    )
    def test_abstract_size(self, xp: ModuleType):
        x = xp.arange(5)
        x = x[x > 2]
        actual = one_hot(x, 5)
        expected = xp.asarray([[0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
        xp_assert_equal(actual, expected)

    @pytest.mark.skip_xp_backend(
        Backend.TORCH_GPU, reason="Puts Pytorch into a bad state."
    )
    def test_out_of_bound(self, xp: ModuleType):
        # Undefined behavior.  Either return zero, or raise.
        try:
            actual = one_hot(xp.asarray([-1, 3]), 3)
        except IndexError:
            return
        expected = xp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        xp_assert_equal(actual, expected)

    @pytest.mark.parametrize(
        "int_dtype",
        ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"],
    )
    def test_int_types(self, xp: ModuleType, int_dtype: str):
        dtype = getattr(xp, int_dtype)
        x = xp.asarray([0, 1, 2], dtype=dtype)
        actual = one_hot(x, 3)
        expected = xp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        xp_assert_equal(actual, expected)

    def test_custom_dtype(self, xp: ModuleType):
        actual = one_hot(xp.asarray([0, 1, 2], dtype=xp.int32), 3, dtype=xp.bool)
        expected = xp.asarray(
            [[True, False, False], [False, True, False], [False, False, True]]
        )
        xp_assert_equal(actual, expected)

    def test_axis(self, xp: ModuleType):
        expected = xp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]).T
        actual = one_hot(xp.asarray([1, 2, 0]), 3, axis=0)
        xp_assert_equal(actual, expected)

        actual = one_hot(xp.asarray([1, 2, 0]), 3, axis=-2)
        xp_assert_equal(actual, expected)

    def test_non_integer(self, xp: ModuleType):
        with pytest.raises(TypeError):
            _ = one_hot(xp.asarray([1.0]), 3)

    def test_device(self, xp: ModuleType, device: Device):
        x = xp.asarray([0, 1, 2], device=device)
        y = one_hot(x, 3)
        assert get_device(y) == device


@pytest.mark.skip_xp_backend(
    Backend.SPARSE, reason="read-only backend without .at support"
)
class TestCreateDiagonal:
    def test_1d_from_numpy(self, xp: ModuleType):
        # from np.diag tests
        vals = 100 * xp.arange(5, dtype=xp.float64)
        b = xp.zeros((5, 5), dtype=xp.float64)
        for k in range(5):
            b = at(b)[k, k].set(vals[k])
        xp_assert_equal(create_diagonal(vals), b)
        b = xp.zeros((7, 7), dtype=xp.float64)
        c = xp.asarray(b, copy=True)
        for k in range(5):
            b = at(b)[k, k + 2].set(vals[k])
            c = at(c)[k + 2, k].set(vals[k])
        xp_assert_equal(create_diagonal(vals, offset=2), b)
        xp_assert_equal(create_diagonal(vals, offset=-2), c)

    @pytest.mark.parametrize("n", range(1, 10))
    @pytest.mark.parametrize("offset", range(1, 10))
    def test_1d_from_scipy(self, xp: ModuleType, n: int, offset: int):
        # from scipy._lib tests
        rng = np.random.default_rng(2347823)
        one = xp.asarray(1.0)
        x = rng.random(n)
        A = create_diagonal(xp.asarray(x, dtype=one.dtype), offset=offset)
        B = xp.asarray(np.diag(x, offset), dtype=one.dtype)
        xp_assert_equal(A, B)

    def test_0d_raises(self, xp: ModuleType):
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
            (4, 2, 1),
            (1, 1, 7),
            (0, 0, 1),
            (3, 2, 4, 5),
        ],
    )
    def test_nd(self, xp: ModuleType, shape: tuple[int, ...]):
        rng = np.random.default_rng(2347823)
        b = xp.asarray(
            rng.integers((1 << 64) - 1, size=shape, dtype=np.uint64), dtype=xp.uint64
        )
        c = create_diagonal(b)
        zero = xp.zeros((), dtype=xp.uint64)
        assert c.shape == (*b.shape, b.shape[-1])
        for i in ndindex(*eager_shape(c)):
            xp_assert_equal(c[i], b[i[:-1]] if i[-2] == i[-1] else zero)

    def test_device(self, xp: ModuleType, device: Device):
        x = xp.asarray([1, 2, 3], device=device)
        assert get_device(create_diagonal(x)) == device

    def test_xp(self, xp: ModuleType):
        x = xp.asarray([1, 2])
        y = create_diagonal(x, xp=xp)
        xp_assert_equal(y, xp.asarray([[1, 0], [0, 2]]))


class TestDefaultDType:
    def test_basic(self, xp: ModuleType):
        assert default_dtype(xp) == xp.empty(0).dtype

    def test_kind(self, xp: ModuleType):
        assert default_dtype(xp, "real floating") == xp.empty(0).dtype
        assert default_dtype(xp, "complex floating") == (xp.empty(0) * 1j).dtype
        assert default_dtype(xp, "integral") == xp.int64
        assert default_dtype(xp, "indexing") == xp.int64

        with pytest.raises(ValueError, match="Unknown kind"):
            _ = default_dtype(xp, "foo")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

    def test_device(self, xp: ModuleType, device: Device):
        # Note: at the moment there are no known namespaces with
        # device-specific default dtypes.
        assert default_dtype(xp, device=None) == xp.empty(0).dtype
        assert default_dtype(xp, device=device) == xp.empty(0).dtype

    def test_torch(self, torch: ModuleType):
        xp = torch
        xp.set_default_dtype(xp.float64)
        assert default_dtype(xp) == xp.float64
        assert default_dtype(xp, "real floating") == xp.float64
        assert default_dtype(xp, "complex floating") == xp.complex128

        xp.set_default_dtype(xp.float32)
        assert default_dtype(xp) == xp.float32
        assert default_dtype(xp, "real floating") == xp.float32
        assert default_dtype(xp, "complex floating") == xp.complex64


class TestExpandDims:
    def test_single_axis(self, xp: ModuleType):
        """Trivial case where xpx.expand_dims doesn't add anything to xp.expand_dims"""
        a = xp.asarray(np.reshape(np.arange(2 * 3 * 4 * 5), (2, 3, 4, 5)))
        for axis in range(-5, 4):
            b = expand_dims(a, axis=axis)
            xp_assert_equal(b, xp.expand_dims(a, axis=axis))

    def test_axis_tuple(self, xp: ModuleType):
        a = xp.empty((3, 3, 3))
        assert expand_dims(a, axis=(0, 1, 2)).shape == (1, 1, 1, 3, 3, 3)
        assert expand_dims(a, axis=(0, -1, -2)).shape == (1, 3, 3, 3, 1, 1)
        assert expand_dims(a, axis=(0, 3, 5)).shape == (1, 3, 3, 1, 3, 1)
        assert expand_dims(a, axis=(0, -3, -5)).shape == (1, 1, 3, 1, 3, 3)

    def test_axis_out_of_range(self, xp: ModuleType):
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

    def test_repeated_axis(self, xp: ModuleType):
        a = xp.empty((3, 3, 3))
        with pytest.raises(ValueError, match="Duplicate dimensions"):
            _ = expand_dims(a, axis=(1, 1))

    def test_positive_negative_repeated(self, xp: ModuleType):
        # https://github.com/data-apis/array-api/issues/760#issuecomment-1989449817
        a = xp.empty((2, 3, 4, 5))
        with pytest.raises(ValueError, match="Duplicate dimensions"):
            _ = expand_dims(a, axis=(3, -3))

    def test_device(self, xp: ModuleType, device: Device):
        x = xp.asarray([1, 2, 3], device=device)
        assert get_device(expand_dims(x, axis=0)) == device

    def test_xp(self, xp: ModuleType):
        x = xp.asarray([1, 2, 3])
        y = expand_dims(x, axis=(0, 1, 2), xp=xp)
        assert y.shape == (1, 1, 1, 3)


@pytest.mark.filterwarnings(  # array_api_strictest
    "ignore:invalid value encountered:RuntimeWarning:array_api_strict"
)
@pytest.mark.filterwarnings(  # sparse
    "ignore:invalid value encountered:RuntimeWarning:sparse"
)
class TestIsClose:
    @pytest.mark.parametrize("swap", [False, True])
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (0.0, 0.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (1.0, -1.0),
            (100.0, 101.0),
            (0, 0),
            (1, 1),
            (1, 2),
            (1, -1),
            (1.0 + 1j, 1.0 + 1j),
            (1.0 + 1j, 1.0 - 1j),
            (float("inf"), float("inf")),
            (float("inf"), 100.0),
            (float("inf"), float("-inf")),
            (float("-inf"), float("-inf")),
            (float("nan"), float("nan")),
            (float("nan"), 100.0),
            (1e6, 1e6 + 1),  # True - within rtol
            (1e6, 1e6 + 100),  # False - outside rtol
            (1e-6, 1.1e-6),  # False - outside atol
            (1e-7, 1.1e-7),  # True - outside atol
            (1e6 + 0j, 1e6 + 1j),  # True - within rtol
            (1e6 + 0j, 1e6 + 100j),  # False - outside rtol
        ],
    )
    def test_basic(self, a: float, b: float, swap: bool, xp: ModuleType):
        if swap:
            b, a = a, b
        a_xp = xp.asarray(a)
        b_xp = xp.asarray(b)

        xp_assert_equal(isclose(a_xp, b_xp), xp.asarray(np.isclose(a, b)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar_np = a * np.arange(10)
            br_np = b * np.arange(10)
            ar_xp = xp.asarray(ar_np)
            br_xp = xp.asarray(br_np)

        xp_assert_equal(isclose(ar_xp, br_xp), xp.asarray(np.isclose(ar_np, br_np)))

    @pytest.mark.parametrize("dtype", ["float32", "int32"])
    def test_broadcast(self, dtype: str, xp: ModuleType):
        dtype = getattr(xp, dtype)
        a = xp.asarray([1, 2, 3], dtype=dtype)
        b = xp.asarray([[1], [5]], dtype=dtype)
        actual = isclose(a, b)
        expect = xp.asarray(
            [[True, False, False], [False, False, False]], dtype=xp.bool
        )

        xp_assert_equal(actual, expect)

    def test_some_inf(self, xp: ModuleType):
        a = xp.asarray([0.0, 1.0, xp.inf, xp.inf, xp.inf])
        b = xp.asarray([1e-9, 1.0, xp.inf, -xp.inf, 2.0])
        actual = isclose(a, b)
        xp_assert_equal(actual, xp.asarray([True, True, True, False, False]))

    def test_equal_nan(self, xp: ModuleType):
        a = xp.asarray([xp.nan, xp.nan, 1.0])
        b = xp.asarray([xp.nan, 1.0, xp.nan])
        xp_assert_equal(isclose(a, b), xp.asarray([False, False, False]))
        xp_assert_equal(isclose(a, b, equal_nan=True), xp.asarray([True, False, False]))

    @pytest.mark.parametrize("dtype", ["float32", "complex64", "int32"])
    def test_tolerance(self, dtype: str, xp: ModuleType):
        dtype = getattr(xp, dtype)
        a = xp.asarray([100, 100], dtype=dtype)
        b = xp.asarray([101, 102], dtype=dtype)
        xp_assert_equal(isclose(a, b), xp.asarray([False, False]))
        xp_assert_equal(isclose(a, b, atol=1), xp.asarray([True, False]))
        xp_assert_equal(isclose(a, b, rtol=0.01), xp.asarray([True, False]))

        # Attempt to trigger division by 0 in rtol on int dtype
        xp_assert_equal(isclose(a, b, rtol=0), xp.asarray([False, False]))
        xp_assert_equal(isclose(a, b, atol=1, rtol=0), xp.asarray([True, False]))

    @pytest.mark.parametrize("dtype", ["int8", "uint8"])
    def test_tolerance_integer_overflow(self, dtype: str, xp: ModuleType):
        """1/rtol is too large for dtype"""
        a = xp.asarray([100, 100], dtype=getattr(xp, dtype))
        b = xp.asarray([100, 101], dtype=getattr(xp, dtype))
        xp_assert_equal(isclose(a, b), xp.asarray([True, False]))

    def test_very_small_numbers(self, xp: ModuleType):
        a = xp.asarray([1e-9, 1e-9])
        b = xp.asarray([1.0001e-9, 1.00001e-9])
        # Difference is below default atol
        xp_assert_equal(isclose(a, b), xp.asarray([True, True]))
        # Use only rtol
        xp_assert_equal(isclose(a, b, atol=0), xp.asarray([False, True]))
        xp_assert_equal(isclose(a, b, atol=0, rtol=0), xp.asarray([False, False]))

    def test_bool_dtype(self, xp: ModuleType):
        a = xp.asarray([False, True, False])
        b = xp.asarray([True, True, False])
        xp_assert_equal(isclose(a, b), xp.asarray([False, True, True]))
        xp_assert_equal(isclose(a, b, atol=1), xp.asarray([True, True, True]))
        xp_assert_equal(isclose(a, b, atol=2), xp.asarray([True, True, True]))
        xp_assert_equal(isclose(a, b, rtol=1), xp.asarray([True, True, True]))
        xp_assert_equal(isclose(a, b, rtol=2), xp.asarray([True, True, True]))

        # Test broadcasting
        xp_assert_equal(
            isclose(a, xp.asarray(True), atol=1), xp.asarray([True, True, True])
        )
        xp_assert_equal(
            isclose(xp.asarray(True), b, atol=1), xp.asarray([True, True, True])
        )

    @pytest.mark.skip_xp_backend(Backend.SPARSE, reason="index by sparse array")
    @pytest.mark.skip_xp_backend(Backend.ARRAY_API_STRICTEST, reason="unknown shape")
    def test_none_shape(self, xp: ModuleType):
        a = xp.asarray([1, 5, 0])
        b = xp.asarray([1, 4, 2])
        b = b[a < 5]
        a = a[a < 5]
        xp_assert_equal(isclose(a, b), xp.asarray([True, False]))

    @pytest.mark.skip_xp_backend(Backend.SPARSE, reason="index by sparse array")
    @pytest.mark.skip_xp_backend(Backend.ARRAY_API_STRICTEST, reason="unknown shape")
    def test_none_shape_bool(self, xp: ModuleType):
        a = xp.asarray([True, True, False])
        b = xp.asarray([True, False, True])
        b = b[a]
        a = a[a]
        xp_assert_equal(isclose(a, b), xp.asarray([True, False]))

    @pytest.mark.skip_xp_backend(Backend.NUMPY_READONLY, reason="xp=xp")
    def test_python_scalar(self, xp: ModuleType):
        a = xp.asarray([0.0, 0.1], dtype=xp.float32)
        xp_assert_equal(isclose(a, 0.0), xp.asarray([True, False]))
        xp_assert_equal(isclose(0.0, a), xp.asarray([True, False]))

        a = xp.asarray([0, 1], dtype=xp.int16)
        xp_assert_equal(isclose(a, 0), xp.asarray([True, False]))
        xp_assert_equal(isclose(0, a), xp.asarray([True, False]))

        xp_assert_equal(isclose(0, 0, xp=xp), xp.asarray(True))
        xp_assert_equal(isclose(0, 1, xp=xp), xp.asarray(False))

    def test_all_python_scalars(self):
        with pytest.raises(TypeError, match="Unrecognized"):
            _ = isclose(0, 0)

    def test_xp(self, xp: ModuleType):
        a = xp.asarray([0.0, 0.0])
        b = xp.asarray([1e-9, 1e-4])
        xp_assert_equal(isclose(a, b, xp=xp), xp.asarray([True, False]))

    @pytest.mark.parametrize("equal_nan", [True, False])
    def test_device(self, xp: ModuleType, device: Device, equal_nan: bool):
        a = xp.asarray([0.0, 0.0, xp.nan], device=device)
        b = xp.asarray([1e-9, 1e-4, xp.nan], device=device)
        res = isclose(a, b, equal_nan=equal_nan)
        assert get_device(res) == device

    def test_array_on_device_with_scalar(self, xp: ModuleType, device: Device):
        a = xp.asarray([0.01, 0.5, 0.8, 0.9, 1.00001], device=device)
        b = 1
        res = isclose(a, b)
        assert get_device(res) == device
        xp_assert_equal(res, xp.asarray([False, False, False, False, True]))

        a = 0.1
        b = xp.asarray([0.01, 0.5, 0.8, 0.9, 0.100001], device=device)
        res = isclose(a, b)
        assert get_device(res) == device
        xp_assert_equal(res, xp.asarray([False, False, False, False, True]))


class TestKron:
    def test_basic(self, xp: ModuleType):
        # Using 0-dimensional array
        a = xp.asarray(1)
        b = xp.asarray([[1, 2], [3, 4]])
        xp_assert_equal(kron(a, b), b)
        xp_assert_equal(kron(b, a), b)

        # Using 1-dimensional array
        a = xp.asarray([3])
        b = xp.asarray([[1, 2], [3, 4]])
        k = xp.asarray([[3, 6], [9, 12]])
        xp_assert_equal(kron(a, b), k)
        xp_assert_equal(kron(b, a), k)

        # Using 3-dimensional array
        a = xp.asarray([[[1]], [[2]]])
        b = xp.asarray([[1, 2], [3, 4]])
        k = xp.asarray([[[1, 2], [3, 4]], [[2, 4], [6, 8]]])
        xp_assert_equal(kron(a, b), k)
        xp_assert_equal(kron(b, a), k)

    def test_kron_smoke(self, xp: ModuleType):
        a = xp.ones((3, 3))
        b = xp.ones((3, 3))
        k = xp.ones((9, 9))

        xp_assert_equal(kron(a, b), k)

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
        self, xp: ModuleType, shape_a: tuple[int, ...], shape_b: tuple[int, ...]
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

    def test_python_scalar(self, xp: ModuleType):
        a = 1
        # Test no dtype promotion to xp.asarray(a); use b.dtype
        b = xp.asarray([[1, 2], [3, 4]], dtype=xp.int16)
        xp_assert_equal(kron(a, b), b)
        xp_assert_equal(kron(b, a), b)
        xp_assert_equal(kron(1, 1, xp=xp), xp.asarray(1))

    def test_all_python_scalars(self):
        with pytest.raises(TypeError, match="Unrecognized"):
            _ = kron(1, 1)

    def test_device(self, xp: ModuleType, device: Device):
        x1 = xp.asarray([1, 2, 3], device=device)
        x2 = xp.asarray([4, 5], device=device)
        assert get_device(kron(x1, x2)) == device

    def test_xp(self, xp: ModuleType):
        a = xp.ones((3, 3))
        b = xp.ones((3, 3))
        k = xp.ones((9, 9))
        xp_assert_equal(kron(a, b, xp=xp), k)


class TestNanToNum:
    def test_bool(self, xp: ModuleType) -> None:
        a = xp.asarray([True])
        xp_assert_equal(nan_to_num(a, xp=xp), a)

    def test_scalar_pos_inf(self, xp: ModuleType, infinity: float) -> None:
        a = xp.inf
        xp_assert_equal(nan_to_num(a, xp=xp), xp.asarray(infinity))

    def test_scalar_neg_inf(self, xp: ModuleType, infinity: float) -> None:
        a = -xp.inf
        xp_assert_equal(nan_to_num(a, xp=xp), -xp.asarray(infinity))

    def test_scalar_nan(self, xp: ModuleType) -> None:
        a = xp.nan
        xp_assert_equal(nan_to_num(a, xp=xp), xp.asarray(0.0))

    def test_real(self, xp: ModuleType, infinity: float) -> None:
        a = xp.asarray([xp.inf, -xp.inf, xp.nan, -128, 128])
        xp_assert_equal(
            nan_to_num(a, xp=xp),
            xp.asarray(
                [
                    infinity,
                    -infinity,
                    0.0,
                    -128,
                    128,
                ]
            ),
        )

    def test_complex(self, xp: ModuleType, infinity: float) -> None:
        a = xp.asarray(
            [
                complex(xp.inf, xp.nan),
                xp.nan,
                complex(xp.nan, xp.inf),
            ]
        )
        xp_assert_equal(
            nan_to_num(a),
            xp.asarray([complex(infinity, 0), complex(0, 0), complex(0, infinity)]),
        )

    def test_empty_array(self, xp: ModuleType) -> None:
        a = xp.asarray([], dtype=xp.float32)  # forced dtype due to torch
        xp_assert_equal(nan_to_num(a, xp=xp), a)
        assert xp.isdtype(nan_to_num(a, xp=xp).dtype, xp.float32)

    @pytest.mark.parametrize(
        ("in_vals", "fill_value", "out_vals"),
        [
            ([1, 2, np.nan, 4], 3, [1.0, 2.0, 3.0, 4.0]),
            ([1, 2, np.nan, 4], 3.0, [1.0, 2.0, 3.0, 4.0]),
            (
                [
                    complex(1, 1),
                    complex(2, 2),
                    complex(np.nan, 0),
                    complex(4, 4),
                ],
                3,
                [
                    complex(1.0, 1.0),
                    complex(2.0, 2.0),
                    complex(3.0, 0.0),
                    complex(4.0, 4.0),
                ],
            ),
            (
                [
                    complex(1, 1),
                    complex(2, 2),
                    complex(0, np.nan),
                    complex(4, 4),
                ],
                3.0,
                [
                    complex(1.0, 1.0),
                    complex(2.0, 2.0),
                    complex(0.0, 3.0),
                    complex(4.0, 4.0),
                ],
            ),
            (
                [
                    complex(1, 1),
                    complex(2, 2),
                    complex(np.nan, np.nan),
                    complex(4, 4),
                ],
                3.0,
                [
                    complex(1.0, 1.0),
                    complex(2.0, 2.0),
                    complex(3.0, 3.0),
                    complex(4.0, 4.0),
                ],
            ),
        ],
    )
    def test_fill_value_success(
        self,
        xp: ModuleType,
        in_vals: Array,
        fill_value: int | float,
        out_vals: Array,
    ) -> None:
        a = xp.asarray(in_vals)
        xp_assert_equal(
            nan_to_num(a, fill_value=fill_value, xp=xp),
            xp.asarray(out_vals),
        )

    def test_fill_value_failure(self, xp: ModuleType) -> None:
        a = xp.asarray(
            [
                complex(1, 1),
                complex(xp.nan, xp.nan),
                complex(3, 3),
            ]
        )
        with pytest.raises(
            TypeError,
            match="Complex fill values are not supported",
        ):
            _ = nan_to_num(
                a,
                fill_value=complex(2, 2),  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
                xp=xp,
            )


class TestNUnique:
    def test_simple(self, xp: ModuleType):
        a = xp.asarray([[1, 1], [0, 2], [2, 2]])
        xp_assert_equal(nunique(a), xp.asarray(3))

    def test_empty(self, xp: ModuleType):
        a = xp.asarray([])
        xp_assert_equal(nunique(a), xp.asarray(0))

    def test_size1(self, xp: ModuleType):
        a = xp.asarray([123])
        xp_assert_equal(nunique(a), xp.asarray(1))

    def test_all_equal(self, xp: ModuleType):
        a = xp.asarray([123, 123, 123])
        xp_assert_equal(nunique(a), xp.asarray(1))

    @pytest.mark.xfail_xp_backend(Backend.DASK, reason="No equal_nan kwarg in unique")
    @pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="sparse#855")
    def test_nan(self, xp: ModuleType, library: Backend):
        if library.like(Backend.NUMPY) and NUMPY_VERSION < (1, 24):
            pytest.xfail("NumPy <1.24 has no equal_nan kwarg in unique")

        # Each NaN is counted separately
        a = xp.asarray([xp.nan, 123.0, xp.nan])
        xp_assert_equal(nunique(a), xp.asarray(3))

    @pytest.mark.parametrize("size", [0, 1, 2])
    def test_device(self, xp: ModuleType, device: Device, size: int):
        a = xp.asarray([0.0] * size, device=device)
        assert get_device(nunique(a)) == device

    def test_xp(self, xp: ModuleType):
        a = xp.asarray([[1, 1], [0, 2], [2, 2]])
        xp_assert_equal(nunique(a, xp=xp), xp.asarray(3))


class TestPad:
    def test_simple(self, xp: ModuleType):
        a = xp.asarray([1, 2, 3])
        padded = pad(a, 2)
        xp_assert_equal(padded, xp.asarray([0, 0, 1, 2, 3, 0, 0]))

    @pytest.mark.xfail_xp_backend(
        Backend.SPARSE, reason="constant_values can only be equal to fill value"
    )
    def test_fill_value(self, xp: ModuleType):
        a = xp.asarray([1, 2, 3])
        padded = pad(a, 2, constant_values=42)
        xp_assert_equal(padded, xp.asarray([42, 42, 1, 2, 3, 42, 42]))

    def test_ndim(self, xp: ModuleType):
        a = xp.asarray(np.reshape(np.arange(2 * 3 * 4), (2, 3, 4)))
        padded = pad(a, 2)
        assert padded.shape == (6, 7, 8)

    def test_mode_not_implemented(self, xp: ModuleType):
        a = xp.asarray([1, 2, 3])
        with pytest.raises(NotImplementedError, match="Only `'constant'`"):
            _ = pad(a, 2, mode="edge")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

    def test_device(self, xp: ModuleType, device: Device):
        a = xp.asarray(0.0, device=device)
        assert get_device(pad(a, 2)) == device

    def test_xp(self, xp: ModuleType):
        padded = pad(xp.asarray(0), 1, xp=xp)
        xp_assert_equal(padded, xp.asarray(0))

    def test_tuple_width(self, xp: ModuleType):
        a = xp.asarray(np.reshape(np.arange(12), (3, 4)))
        padded = pad(a, (1, 0))
        assert padded.shape == (4, 5)

        padded = pad(a, (1, 2))
        assert padded.shape == (6, 7)

        with pytest.raises((ValueError, RuntimeError)):
            _ = pad(a, [(1, 2, 3)])  # type: ignore[list-item]  # pyright: ignore[reportArgumentType]

    def test_sequence_of_tuples_width(self, xp: ModuleType):
        a = xp.asarray(np.reshape(np.arange(12), (3, 4)))

        padded = pad(a, ((1, 0), (0, 2)))
        assert padded.shape == (4, 6)
        padded = pad(a, ((1, 0), (0, 0)))
        assert padded.shape == (4, 4)


assume_unique = pytest.mark.parametrize(
    "assume_unique",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail_xp_backend(
                Backend.DASK, reason="NaN-shaped arrays"
            ),
        ),
    ],
)


@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no argsort")
@pytest.mark.skip_xp_backend(Backend.ARRAY_API_STRICTEST, reason="no unique_values")
class TestSetDiff1D:
    @pytest.mark.xfail_xp_backend(Backend.DASK, reason="NaN-shaped arrays")
    @pytest.mark.xfail_xp_backend(
        Backend.TORCH, reason="index_select not implemented for uint32"
    )
    @pytest.mark.xfail_xp_backend(
        Backend.TORCH_GPU, reason="index_select not implemented for uint32"
    )
    def test_setdiff1d(self, xp: ModuleType):
        x1 = xp.asarray([6, 5, 4, 7, 1, 2, 7, 4])
        x2 = xp.asarray([2, 4, 3, 3, 2, 1, 5])

        expected = xp.asarray([6, 7])
        actual = setdiff1d(x1, x2)
        xp_assert_equal(actual, expected)

        x1 = xp.arange(21)
        x2 = xp.arange(19)
        expected = xp.asarray([19, 20])
        actual = setdiff1d(x1, x2)
        xp_assert_equal(actual, expected)

        xp_assert_equal(setdiff1d(xp.empty(0), xp.empty(0)), xp.empty(0))
        x1 = xp.empty(0, dtype=xp.uint32)
        x2 = x1
        assert xp.isdtype(setdiff1d(x1, x2).dtype, xp.uint32)

    def test_assume_unique(self, xp: ModuleType):
        x1 = xp.asarray([3, 2, 1])
        x2 = xp.asarray([7, 5, 2])
        expected = xp.asarray([3, 1])
        actual = setdiff1d(x1, x2, assume_unique=True)
        xp_assert_equal(actual, expected)

    @assume_unique
    @pytest.mark.parametrize("shape1", [(), (1,), (1, 1)])
    @pytest.mark.parametrize("shape2", [(), (1,), (1, 1)])
    def test_shapes(
        self,
        assume_unique: bool,
        shape1: tuple[int, ...],
        shape2: tuple[int, ...],
        xp: ModuleType,
    ):
        x1 = xp.zeros(shape1)
        x2 = xp.zeros(shape2)
        actual = setdiff1d(x1, x2, assume_unique=assume_unique)
        xp_assert_equal(actual, xp.empty((0,)))

    @assume_unique
    @pytest.mark.skip_xp_backend(Backend.NUMPY_READONLY, reason="xp=xp")
    def test_python_scalar(self, xp: ModuleType, assume_unique: bool):
        # Test no dtype promotion to xp.asarray(x2); use x1.dtype
        x1 = xp.asarray([3, 1, 2], dtype=xp.int16)
        x2 = 3
        actual = setdiff1d(x1, x2, assume_unique=assume_unique)
        xp_assert_equal(actual, xp.asarray([1, 2], dtype=xp.int16))

        actual = setdiff1d(x2, x1, assume_unique=assume_unique)
        xp_assert_equal(actual, xp.asarray([], dtype=xp.int16))

        xp_assert_equal(
            setdiff1d(0, 0, assume_unique=assume_unique, xp=xp),
            xp.asarray([0])[:0],  # Default int dtype for backend
        )

    @pytest.mark.parametrize("assume_unique", [True, False])
    def test_all_python_scalars(self, assume_unique: bool):
        with pytest.raises(TypeError, match="Unrecognized"):
            _ = setdiff1d(0, 0, assume_unique=assume_unique)

    @assume_unique
    @pytest.mark.skip_xp_backend(
        Backend.TORCH, reason="device='meta' does not support unknown shapes"
    )
    def test_device(self, xp: ModuleType, device: Device, assume_unique: bool):
        x1 = xp.asarray([3, 8, 20], device=device)
        x2 = xp.asarray([2, 3, 4], device=device)
        assert get_device(setdiff1d(x1, x2, assume_unique=assume_unique)) == device

    @pytest.mark.skip_xp_backend(Backend.NUMPY_READONLY, reason="xp=xp")
    def test_xp(self, xp: ModuleType):
        x1 = xp.asarray([3, 8, 20])
        x2 = xp.asarray([2, 3, 4])
        expected = xp.asarray([8, 20])
        actual = setdiff1d(x1, x2, assume_unique=True, xp=xp)
        xp_assert_equal(actual, expected)


class TestSinc:
    def test_simple(self, xp: ModuleType):
        xp_assert_equal(sinc(xp.asarray(0.0)), xp.asarray(1.0))
        x = xp.asarray(np.linspace(-1, 1, 100))
        w = sinc(x)
        # check symmetry
        xp_assert_close(w, xp.flip(w, axis=0))

    @pytest.mark.parametrize("x", [0, 1 + 3j])
    def test_dtype(self, xp: ModuleType, x: int | complex):
        with pytest.raises(ValueError, match="real floating data type"):
            _ = sinc(xp.asarray(x))

    def test_3d(self, xp: ModuleType):
        x = np.arange(18, dtype=np.float64).reshape((3, 3, 2))
        expected = np.zeros_like(x)
        expected[0, 0, 0] = 1
        x = xp.asarray(x)
        expected = xp.asarray(expected)
        xp_assert_close(sinc(x), expected, atol=1e-15)

    def test_device(self, xp: ModuleType, device: Device):
        x = xp.asarray(0.0, device=device)
        assert get_device(sinc(x)) == device

    def test_xp(self, xp: ModuleType):
        xp_assert_equal(sinc(xp.asarray(0.0), xp=xp), xp.asarray(1.0))


class TestPartition:
    @classmethod
    def _assert_valid_partition(
        cls,
        x_np: np.ndarray | None,
        k: int,
        y: Array,
        xp: ModuleType,
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
    def _partition(cls, x: np.ndarray, k: int, xp: ModuleType, axis: int | None = -1):
        return partition(xp.asarray(x), k, axis=axis)

    def _test_1d(self, xp: ModuleType):
        rng = np.random.default_rng()
        for n in [2, 3, 4, 5, 7, 10, 20, 50, 100, 1_000]:
            k = int(rng.integers(n))
            x1 = rng.integers(n, size=n)
            y = self._partition(x1, k, xp)
            self._assert_valid_partition(x1, k, y, xp)
            x2 = rng.random(n)
            y = self._partition(x2, k, xp)
            self._assert_valid_partition(x2, k, y, xp)

    def _test_nd(self, xp: ModuleType, ndim: int):
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

    def _test_input_validation(self, xp: ModuleType):
        with pytest.raises(TypeError):
            _ = self._partition(np.asarray(1), 1, xp)
        with pytest.raises(ValueError, match="out of bounds"):
            _ = self._partition(np.asarray([1, 2]), 3, xp)

    def test_1d(self, xp: ModuleType):
        self._test_1d(xp)

    @pytest.mark.parametrize("ndim", [2, 3, 4])
    def test_nd(self, xp: ModuleType, ndim: int):
        self._test_nd(xp, ndim)

    def test_input_validation(self, xp: ModuleType):
        self._test_input_validation(xp)


@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no argsort")
class TestArgpartition(TestPartition):
    @classmethod
    @override
    def _partition(cls, x: np.ndarray, k: int, xp: ModuleType, axis: int | None = -1):
        arr = xp.asarray(x)
        indices = argpartition(arr, k, axis=axis)
        if axis is None:
            arr = xp.reshape(arr, shape=(-1,))
            return arr[indices]
        if arr.ndim == 1:
            return arr[indices]
        return cls._take_along_axis(arr, indices, axis=axis, xp=xp)

    @classmethod
    def _take_along_axis(cls, arr: Array, indices: Array, axis: int, xp: ModuleType):
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
    def test_1d(self, xp: ModuleType):
        self._test_1d(xp)

    @pytest.mark.parametrize("ndim", [2, 3, 4])
    @override
    def test_nd(self, xp: ModuleType, ndim: int):
        self._test_nd(xp, ndim)

    @override
    def test_input_validation(self, xp: ModuleType):
        self._test_input_validation(xp)


@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no unique_inverse")
class TestIsIn:
    def test_simple(self, xp: ModuleType, library: Backend):
        if library.like(Backend.NUMPY) and NUMPY_VERSION < (1, 24):
            pytest.xfail("NumPy <1.24 has no kind kwarg in isin")

        b = xp.asarray([1, 2, 3, 4])

        # `a` with 1 dimension
        a = xp.asarray([1, 3, 6, 10])
        expected = xp.asarray([True, True, False, False])
        res = isin(a, b)
        xp_assert_equal(res, expected)

        # `a` with 2 dimensions
        a = xp.asarray([[0, 2], [4, 6]])
        expected = xp.asarray([[False, True], [True, False]])
        res = isin(a, b)
        xp_assert_equal(res, expected)

    def test_device(self, xp: ModuleType, device: Device, library: Backend):
        if library.like(Backend.NUMPY) and NUMPY_VERSION < (1, 24):
            pytest.xfail("NumPy <1.24 has no kind kwarg in isin")

        a = xp.asarray([1, 3, 6], device=device)
        b = xp.asarray([1, 2, 3], device=device)
        assert get_device(isin(a, b)) == device

    def test_assume_unique_and_invert(
        self, xp: ModuleType, device: Device, library: Backend
    ):
        if library.like(Backend.NUMPY) and NUMPY_VERSION < (1, 24):
            pytest.xfail("NumPy <1.24 has no kind kwarg in isin")

        a = xp.asarray([0, 3, 6, 10], device=device)
        b = xp.asarray([1, 2, 3, 10], device=device)
        expected = xp.asarray([True, False, True, False])
        res = isin(a, b, assume_unique=True, invert=True)
        assert get_device(res) == device
        xp_assert_equal(res, expected)

    def test_kind(self, xp: ModuleType, library: Backend):
        if library.like(Backend.NUMPY) and NUMPY_VERSION < (1, 24):
            pytest.xfail("NumPy <1.24 has no kind kwarg in isin")

        a = xp.asarray([0, 3, 6, 10])
        b = xp.asarray([1, 2, 3, 10])
        expected = xp.asarray([False, True, False, True])
        res = isin(a, b, kind="sort")
        xp_assert_equal(res, expected)


METHOD = Literal["linear", "inverted_cdf", "averaged_inverted_cdf"]


@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no xp.take")
class TestQuantile:
    def test_basic(self, xp: ModuleType):
        x = xp.asarray([1, 2, 3, 4, 5])
        actual = quantile(x, 0.5)
        expect = xp.asarray(3.0, dtype=xp.float64)
        xp_assert_close(actual, expect)

    def test_xp(self, xp: ModuleType):
        x = xp.asarray([1, 2, 3, 4, 5])
        actual = quantile(x, 0.5, xp=xp)
        expect = xp.asarray(3.0, dtype=xp.float64)
        xp_assert_close(actual, expect)

    def test_multiple_quantiles(self, xp: ModuleType):
        x = xp.asarray([1, 2, 3, 4, 5])
        actual = quantile(x, xp.asarray([0.25, 0.5, 0.75]))
        expect = xp.asarray([2.0, 3.0, 4.0], dtype=xp.float64)
        xp_assert_close(actual, expect)

    def test_shape(self, xp: ModuleType):
        rng = np.random.default_rng()
        a = xp.asarray(rng.random((3, 4, 5)))
        q = xp.asarray(rng.random(2))
        assert quantile(a, q, axis=0).shape == (2, 4, 5)
        assert quantile(a, q, axis=1).shape == (2, 3, 5)
        assert quantile(a, q, axis=2).shape == (2, 3, 4)

        assert quantile(a, q, axis=0, keepdims=True).shape == (2, 1, 4, 5)
        assert quantile(a, q, axis=1, keepdims=True).shape == (2, 3, 1, 5)
        assert quantile(a, q, axis=2, keepdims=True).shape == (2, 3, 4, 1)

    @pytest.mark.parametrize("with_nans", ["no_nans", "with_nans"])
    @pytest.mark.parametrize("method", get_args(METHOD))
    def test_against_numpy_1d(self, xp: ModuleType, with_nans: str, method: METHOD):
        rng = np.random.default_rng()
        a_np = rng.random(40)
        if with_nans == "with_nans":
            a_np[rng.random(a_np.shape) < rng.random() * 0.5] = np.nan
        q_np = np.asarray([0, *rng.random(2), 1])
        a = xp.asarray(a_np)
        q = xp.asarray(q_np)

        actual = quantile(a, q, method=method)
        expected = np.quantile(a_np, q_np, method=method)
        expected = xp.asarray(expected)
        xp_assert_close(actual, expected)

    @pytest.mark.parametrize("with_nans", ["no_nans", "with_nans"])
    @pytest.mark.parametrize("method", get_args(METHOD))
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_against_numpy_nd(
        self, xp: ModuleType, keepdims: bool, with_nans: str, method: METHOD
    ):
        rng = np.random.default_rng()
        a_np = rng.random((3, 4, 5))
        if with_nans == "with_nans":
            a_np[rng.random(a_np.shape) < rng.random()] = np.nan
        q_np = rng.random(2)
        a = xp.asarray(a_np)
        q = xp.asarray(q_np)
        for axis in [None, *range(a.ndim)]:
            actual = quantile(a, q, axis=axis, keepdims=keepdims, method=method)
            expected = np.quantile(
                a_np, q_np, axis=axis, keepdims=keepdims, method=method
            )
            expected = xp.asarray(expected)
            xp_assert_close(actual, expected)

    @pytest.mark.parametrize("nan_policy", ["no_nans", "propagate"])
    @pytest.mark.parametrize("with_weights", ["with_weights", "no_weights"])
    def test_against_median_min_max(
        self,
        xp: ModuleType,
        nan_policy: str,
        with_weights: str,
    ):
        rng = np.random.default_rng()
        n = 40
        a_np = rng.random(n)
        w_np = rng.integers(0, 2, n) if with_weights == "with_weights" else None
        if nan_policy == "no_nans":
            nan_policy = "propagate"
        else:
            # from 0% to 50% of NaNs:
            a_np[rng.random(n) < rng.random(n) * 0.5] = np.nan
            if w_np is not None:
                # ensure at least one NaN on non-null weight:
                (nz_weights_idx,) = np.where(w_np > 0)
                a_np[nz_weights_idx[0]] = np.nan

        a_np_med = a_np if w_np is None else a_np[w_np > 0]
        a = xp.asarray(a_np)
        w = xp.asarray(w_np) if w_np is not None else None

        np_median = np.nanmedian if nan_policy == "omit" else np.median
        expected = np_median(a_np_med)
        method = "averaged_inverted_cdf"
        actual = quantile(a, 0.5, method=method, nan_policy=nan_policy, weights=w)
        xp_assert_close(actual, xp.asarray(expected))

        for method in ["inverted_cdf", "averaged_inverted_cdf"]:
            np_min = np.nanmin if nan_policy == "omit" else np.min
            expected = np_min(a_np_med)
            actual = quantile(a, 0.0, method=method, nan_policy=nan_policy, weights=w)
            xp_assert_close(actual, xp.asarray(expected))

            np_max = np.nanmax if nan_policy == "omit" else np.max
            expected = np_max(a_np_med)
            actual = quantile(a, 1.0, method=method, nan_policy=nan_policy, weights=w)
            xp_assert_close(actual, xp.asarray(expected))

    @pytest.mark.parametrize("keepdims", [True, False])
    @pytest.mark.parametrize("nan_policy", ["no_nans", "propagate", "omit"])
    @pytest.mark.parametrize("q_np", [0.5, 0.0, 1.0, np.linspace(0, 1, num=11)])
    def test_weighted_against_numpy(
        self, xp: ModuleType, keepdims: bool, q_np: Array | float, nan_policy: str
    ):
        if NUMPY_VERSION < (2, 0):
            pytest.xfail(reason="NumPy 1.x does not support weights in quantile")
        rng = np.random.default_rng()
        n, d = 10, 20
        a_2d = rng.random((n, d))
        mask_nan = np.zeros((n, d), dtype=bool)
        if nan_policy == "no_nans":
            nan_policy = "propagate"
        else:
            # from 0% to 100% of NaNs:
            mask_nan = rng.random((n, d)) < rng.random((n, 1))
            # don't put nans in the first row:
            mask_nan[:] = False
            a_2d[mask_nan] = np.nan

        q = xp.asarray(q_np, copy=True)
        m: METHOD = "inverted_cdf"

        np_quantile = np.quantile
        if nan_policy == "omit":
            np_quantile = np.nanquantile

        for a_np, w_np, axis in [
            (a_2d, rng.random(n), 0),
            (a_2d, rng.random(d), 1),
            (a_2d[0], rng.random(d), None),
            (a_2d, rng.integers(0, 3, n), 0),
            (a_2d, rng.integers(0, 2, d), 1),
            (a_2d, rng.integers(0, 2, (n, d)), 0),
            (a_2d, rng.integers(0, 3, (n, d)), 1),
        ]:
            with warnings.catch_warnings(record=True) as warning:
                divide_msg = "invalid value encountered in divide"
                warnings.filterwarnings("always", divide_msg, RuntimeWarning)
                nan_slice_msg = "All-NaN slice encountered"
                warnings.filterwarnings("ignore", nan_slice_msg, RuntimeWarning)
                try:
                    expected = np_quantile(
                        a_np,
                        np.asarray(q_np),
                        axis=axis,
                        method=m,
                        weights=w_np,  # type: ignore[arg-type]
                        keepdims=keepdims,
                    )
                except IndexError:
                    continue
                if warning:
                    # this means some weights sum was 0
                    # in this case we skip calling xpx.quantile
                    continue
            expected = xp.asarray(expected)

            a = xp.asarray(a_np)
            w = xp.asarray(w_np)
            actual = quantile(
                a,
                q,
                axis=axis,
                method=m,
                weights=w,
                keepdims=keepdims,
                nan_policy=nan_policy,
            )
            xp_assert_close(actual, expected)

    def test_methods(self, xp: ModuleType):
        x = xp.asarray([1, 2, 3, 4, 5])
        methods = ["linear", "inverted_cdf", "averaged_inverted_cdf"]
        for method in methods:
            actual = quantile(x, 0.5, method=method)
            # All methods should give reasonable results
            assert 2.5 <= float(actual) <= 3.5

    def test_edge_cases(self, xp: ModuleType):
        x = xp.asarray([1, 2, 3, 4, 5])
        # q = 0 should give minimum
        actual = quantile(x, 0.0)
        expect = xp.asarray(1.0, dtype=xp.float64)
        xp_assert_close(actual, expect)

        # q = 1 should give maximum
        actual = quantile(x, 1.0)
        expect = xp.asarray(5.0, dtype=xp.float64)
        xp_assert_close(actual, expect)

    def test_invalid_q(self, xp: ModuleType):
        x = xp.asarray([1, 2, 3, 4, 5])
        # q > 1 should raise
        with pytest.raises(
            ValueError, match=r"`q` values must be in the range \[0, 1\]"
        ):
            _ = quantile(x, 1.5)
        # q < 0 should raise
        with pytest.raises(
            ValueError, match=r"`q` values must be in the range \[0, 1\]"
        ):
            _ = quantile(x, -0.5)

    def test_invalid_shape(self, xp: ModuleType):
        with pytest.raises(TypeError, match="at least 1-dimensional"):
            _ = quantile(xp.asarray(3.0), 0.5)
        with pytest.raises(ValueError, match="not compatible with the dimension"):
            _ = quantile(xp.asarray([3.0]), 0.5, axis=1)
        # with weights:
        method = "inverted_cdf"

        shape = (2, 3, 4)
        with pytest.raises(ValueError, match="dimension of `a` must be 1 or 2"):
            _ = quantile(
                xp.ones(shape), 0.5, axis=1, weights=xp.ones(shape), method=method
            )

        with pytest.raises(TypeError, match="Axis must be specified"):
            _ = quantile(xp.ones((2, 3)), 0.5, weights=xp.ones(3), method=method)

        with pytest.raises(ValueError, match="Shape of weights must be consistent"):
            _ = quantile(
                xp.ones((2, 3)), 0.5, axis=0, weights=xp.ones(3), method=method
            )

        with pytest.raises(ValueError, match="Axis must be specified"):
            _ = quantile(xp.ones((2, 3)), 0.5, weights=xp.ones((2, 3)), method=method)

    def test_invalid_dtype(self, xp: ModuleType):
        with pytest.raises(ValueError, match="`a` must have real dtype"):
            _ = quantile(xp.ones(5, dtype=xp.bool), 0.5)

        a = xp.ones(5)
        with pytest.raises(ValueError, match="`q` must have real floating dtype"):
            _ = quantile(a, xp.asarray([0, 1]))

        weights = xp.ones(5, dtype=xp.bool)
        with pytest.raises(ValueError, match="`weights` must have real dtype"):
            _ = quantile(a, 0.5, weights=weights, method="inverted_cdf")

    def test_invalid_method(self, xp: ModuleType):
        with pytest.raises(ValueError, match="`method` must be one of"):
            _ = quantile(xp.ones(5), 0.5, method="invalid")

        with pytest.raises(ValueError, match="not supported with weights"):
            _ = quantile(xp.ones(5), 0.5, method="linear", weights=xp.ones(5))

    def test_invalid_nan_policy(self, xp: ModuleType):
        with pytest.raises(ValueError, match="`nan_policy` must be one of"):
            _ = quantile(xp.ones(5), 0.5, nan_policy="invalid")

        with pytest.raises(ValueError, match="must be 'propagate'"):
            _ = quantile(xp.ones(5), 0.5, nan_policy="omit")

    def test_device(self, xp: ModuleType, device: Device):
        if hasattr(device, "type") and device.type == "meta":  # pyright: ignore[reportAttributeAccessIssue]
            pytest.xfail("No Tensor.item() on meta device")
        x = xp.asarray([1, 2, 3, 4, 5], device=device)
        actual = quantile(x, 0.5)
        assert get_device(actual) == device
