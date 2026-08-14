import warnings
from typing import Any, cast

import hypothesis
import hypothesis.extra.numpy as npst
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from array_api_extra import (
    angle,
    apply_where,
    default_dtype,
    deg2rad,
    isclose,
    nan_to_num,
    rad2deg,
    sinc,
)
from array_api_extra._lib._backends import NUMPY_VERSION, Backend
from array_api_extra._lib._compat import device as get_device
from array_api_extra._lib._typing import Array, ArrayNamespace, Device
from array_api_extra.testing import assert_close, assert_equal, lazy_xp_function

lazy_xp_function(apply_where)
lazy_xp_function(deg2rad)
lazy_xp_function(isclose)
lazy_xp_function(nan_to_num)
lazy_xp_function(rad2deg)
lazy_xp_function(sinc)


class TestApplyWhere:
    @staticmethod
    def f1(x: Array, y: Array | int = 10) -> Array:
        return x + y

    @staticmethod
    def f2(x: Array, y: Array | int = 10) -> Array:
        return x - y

    def test_f1_f2(self, xp: ArrayNamespace):
        x = xp.asarray([1, 2, 3, 4])
        cond = x % 2 == 0
        actual = apply_where(cond, x, self.f1, self.f2)
        expect = xp.where(cond, self.f1(x), self.f2(x))
        assert_equal(actual, expect)

    def test_fill_value(self, xp: ArrayNamespace):
        x = xp.asarray([1, 2, 3, 4])
        cond = x % 2 == 0
        actual = apply_where(x % 2 == 0, x, self.f1, fill_value=0)
        expect = xp.where(cond, self.f1(x), xp.asarray(0))
        assert_equal(actual, expect)

        actual = apply_where(x % 2 == 0, x, self.f1, fill_value=xp.asarray(0))
        assert_equal(actual, expect)

    def test_args_tuple(self, xp: ArrayNamespace):
        x = xp.asarray([1, 2, 3, 4])
        y = xp.asarray([10, 20, 30, 40])
        cond = x % 2 == 0
        actual = apply_where(cond, (x, y), self.f1, self.f2)
        expect = xp.where(cond, self.f1(x, y), self.f2(x, y))
        assert_equal(actual, expect)

    def test_broadcast(self, xp: ArrayNamespace):
        x = xp.asarray([1, 2])
        y = xp.asarray([[10], [20], [30]])
        cond = xp.broadcast_to(xp.asarray(True), (4, 1, 1))

        actual = apply_where(cond, (x, y), self.f1, self.f2)
        expect = xp.where(cond, self.f1(x, y), self.f2(x, y))
        assert_equal(actual, expect)

        actual = apply_where(
            cond,
            (x, y),
            lambda x, _: x,
            lambda _, y: y,
        )
        expect = xp.where(cond, x, y)
        assert_equal(actual, expect)

        # Shaped fill_value
        actual = apply_where(cond, x, self.f1, fill_value=y)
        expect = xp.where(cond, self.f1(x), y)
        assert_equal(actual, expect)

    def test_dtype_propagation(self, xp: ArrayNamespace, library: Backend):
        x = xp.asarray([1, 2], dtype=xp.int8)
        y = xp.asarray([3, 4], dtype=xp.int16)
        cond = x % 2 == 0

        mxp = np if library is Backend.DASK else xp
        actual = apply_where(
            cond,
            (x, y),
            self.f1,
            lambda x, y: mxp.astype(x - y, xp.int64),  # pyright: ignore[reportArgumentType] # pyrefly: ignore[bad-argument-type]
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
        xp: ArrayNamespace,
        fill_value_raw: int | list[int],
        fill_value_dtype: str,
        expect_dtype: str,
    ):
        x = xp.asarray([1, 2], dtype=xp.int16)
        cond = x % 2 == 0
        fill_value = xp.asarray(fill_value_raw, dtype=getattr(xp, fill_value_dtype))

        actual = apply_where(cond, x, self.f1, fill_value=fill_value)
        assert actual.dtype == getattr(xp, expect_dtype)

    def test_dont_overwrite_fill_value(self, xp: ArrayNamespace):
        x = xp.asarray([1, 2])
        fill_value = xp.asarray([100, 200])
        actual = apply_where(x % 2 == 0, x, self.f1, fill_value=fill_value)
        assert_equal(actual, xp.asarray([100, 12]))
        assert_equal(fill_value, xp.asarray([100, 200]))

    @pytest.mark.skip_xp_backend(
        Backend.ARRAY_API_STRICTEST,
        reason="no boolean indexing -> run everywhere",
    )
    @pytest.mark.skip_xp_backend(
        Backend.SPARSE,
        reason="no indexing by sparse array -> run everywhere",
    )
    def test_dont_run_on_false(self, xp: ArrayNamespace):
        x = xp.asarray([1.0, 2.0, 0.0])
        y = xp.asarray([0.0, 3.0, 4.0])
        # On NumPy, division by zero will trigger warnings
        actual = apply_where(
            x == 0,
            (x, y),
            lambda x, y: x / y,
            lambda x, y: y / x,
        )
        assert_equal(actual, xp.asarray([0.0, 1.5, 0.0]))

    def test_bad_args(self, xp: ArrayNamespace):
        x = xp.asarray([1, 2, 3, 4])
        cond = x % 2 == 0
        # Neither f2 nor fill_value
        with pytest.raises(TypeError, match="Exactly one of"):
            apply_where(cond, x, self.f1)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue]
        # Both f2 and fill_value
        with pytest.raises(TypeError, match="Exactly one of"):
            apply_where(cond, x, self.f1, self.f2, fill_value=0)  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue]

    @pytest.mark.skip_xp_backend(Backend.NUMPY_READONLY, reason="xp=xp")
    def test_xp(self, xp: ArrayNamespace):
        x = xp.asarray([1, 2, 3, 4])
        cond = x % 2 == 0
        actual = apply_where(cond, x, self.f1, self.f2, xp=xp)
        expect = xp.where(cond, self.f1(x), self.f2(x))
        assert_equal(actual, expect)

    def test_device(self, xp: ArrayNamespace, device: Device):
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
        n_arrays=st.integers(min_value=0, max_value=3),
        n_kwarrays=st.integers(min_value=0, max_value=3),
        rng_seed=st.integers(min_value=1000000000, max_value=9999999999),
        dtype=npst.floating_dtypes(sizes=(32, 64)),
        p=st.floats(min_value=0, max_value=1),
        data=st.data(),
    )
    def test_hypothesis(
        self,
        n_arrays: int,
        n_kwarrays: int,
        rng_seed: int,
        dtype: np.dtype[Any],
        p: float,
        data: st.DataObject,
        xp: ArrayNamespace,
        library: Backend,
    ):
        if (
            library.like(Backend.NUMPY)
            and NUMPY_VERSION < (2, 0)
            and dtype.type is np.float32
        ):
            pytest.xfail(reason="NumPy 1.x dtype promotion for scalars")

        _ = hypothesis.assume(n_arrays + n_kwarrays > 0)
        mbs = npst.mutually_broadcastable_shapes(
            num_shapes=1 + n_arrays + n_kwarrays, min_side=0
        )
        input_shapes, _ = data.draw(mbs)
        cond_shape = input_shapes[0]
        shapes = input_shapes[1 : 1 + n_arrays]
        kwshapes = input_shapes[1 + n_arrays :]

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

        kwargs = {
            f"kw{n}": xp.asarray(
                data.draw(npst.arrays(dtype=dtype.type, shape=shape, elements=elements))
            )
            for n, shape in enumerate(kwshapes)
        }
        kwkeys = kwargs.keys()

        def f1(*args: Array, **kwargs: dict[str, Array]) -> Array:
            assert kwargs.keys() == kwkeys
            args_kwargs = cast(tuple[Array, ...], (*args, *kwargs.values()))
            return cast(Array, sum(args_kwargs))

        def f2(*args: Array, **kwargs: dict[str, Array]) -> Array:
            assert kwargs.keys() == kwkeys
            args_kwargs = cast(tuple[Array, ...], (*args, *kwargs.values()))
            return cast(Array, sum(args_kwargs) / 2)

        rng = np.random.default_rng(rng_seed)
        cond = xp.asarray(rng.random(size=cond_shape) > p)

        res1 = apply_where(cond, arrays, f1, fill_value=fill_value, kwargs=kwargs)
        res2 = apply_where(cond, arrays, f1, f2, kwargs=kwargs)
        res3 = apply_where(cond, arrays, f1, fill_value=float_fill_value, kwargs=kwargs)

        ref1 = xp.where(cond, f1(*arrays, **kwargs), fill_value)
        ref2 = xp.where(cond, f1(*arrays, **kwargs), f2(*arrays, **kwargs))
        ref3 = xp.where(cond, f1(*arrays, **kwargs), float_fill_value)

        assert_close(res1, ref1, rtol=2e-16)
        assert_equal(res2, ref2)
        assert_equal(res3, ref3)


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
    def test_basic(self, a: float, b: float, swap: bool, xp: ArrayNamespace):
        if swap:
            b, a = a, b
        a_xp = xp.asarray(a)
        b_xp = xp.asarray(b)

        assert_equal(isclose(a_xp, b_xp), xp.asarray(np.isclose(a, b)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ar_np = a * np.arange(10)
            br_np = b * np.arange(10)
            ar_xp = xp.asarray(ar_np)
            br_xp = xp.asarray(br_np)

        assert_equal(isclose(ar_xp, br_xp), xp.asarray(np.isclose(ar_np, br_np)))

    @pytest.mark.parametrize("dtype", ["float32", "int32"])
    def test_broadcast(self, dtype: str, xp: ArrayNamespace):
        dtype = getattr(xp, dtype)
        a = xp.asarray([1, 2, 3], dtype=dtype)
        b = xp.asarray([[1], [5]], dtype=dtype)
        actual = isclose(a, b)
        expect = xp.asarray(
            [[True, False, False], [False, False, False]], dtype=xp.bool
        )

        assert_equal(actual, expect)

    def test_some_inf(self, xp: ArrayNamespace):
        a = xp.asarray([0.0, 1.0, xp.inf, xp.inf, xp.inf])
        b = xp.asarray([1e-9, 1.0, xp.inf, -xp.inf, 2.0])
        actual = isclose(a, b)
        assert_equal(actual, xp.asarray([True, True, True, False, False]))

    def test_equal_nan(self, xp: ArrayNamespace):
        a = xp.asarray([xp.nan, xp.nan, 1.0])
        b = xp.asarray([xp.nan, 1.0, xp.nan])
        assert_equal(isclose(a, b), xp.asarray([False, False, False]))
        assert_equal(isclose(a, b, equal_nan=True), xp.asarray([True, False, False]))

    @pytest.mark.parametrize("dtype", ["float32", "complex64", "int32"])
    def test_tolerance(self, dtype: str, xp: ArrayNamespace):
        dtype = getattr(xp, dtype)
        a = xp.asarray([100, 100], dtype=dtype)
        b = xp.asarray([101, 102], dtype=dtype)
        assert_equal(isclose(a, b), xp.asarray([False, False]))
        assert_equal(isclose(a, b, atol=1), xp.asarray([True, False]))
        assert_equal(isclose(a, b, rtol=0.01), xp.asarray([True, False]))

        # Attempt to trigger division by 0 in rtol on int dtype
        assert_equal(isclose(a, b, rtol=0), xp.asarray([False, False]))
        assert_equal(isclose(a, b, atol=1, rtol=0), xp.asarray([True, False]))

    @pytest.mark.parametrize("dtype", ["int8", "uint8"])
    def test_tolerance_integer_overflow(self, dtype: str, xp: ArrayNamespace):
        """1/rtol is too large for dtype"""
        a = xp.asarray([100, 100], dtype=getattr(xp, dtype))
        b = xp.asarray([100, 101], dtype=getattr(xp, dtype))
        assert_equal(isclose(a, b), xp.asarray([True, False]))

    def test_very_small_numbers(self, xp: ArrayNamespace):
        a = xp.asarray([1e-9, 1e-9])
        b = xp.asarray([1.0001e-9, 1.00001e-9])
        # Difference is below default atol
        assert_equal(isclose(a, b), xp.asarray([True, True]))
        # Use only rtol
        assert_equal(isclose(a, b, atol=0), xp.asarray([False, True]))
        assert_equal(isclose(a, b, atol=0, rtol=0), xp.asarray([False, False]))

    def test_bool_dtype(self, xp: ArrayNamespace):
        a = xp.asarray([False, True, False])
        b = xp.asarray([True, True, False])
        assert_equal(isclose(a, b), xp.asarray([False, True, True]))
        assert_equal(isclose(a, b, atol=1), xp.asarray([True, True, True]))
        assert_equal(isclose(a, b, atol=2), xp.asarray([True, True, True]))
        assert_equal(isclose(a, b, rtol=1), xp.asarray([True, True, True]))
        assert_equal(isclose(a, b, rtol=2), xp.asarray([True, True, True]))

        # Test broadcasting
        assert_equal(
            isclose(a, xp.asarray(True), atol=1), xp.asarray([True, True, True])
        )
        assert_equal(
            isclose(xp.asarray(True), b, atol=1), xp.asarray([True, True, True])
        )

    @pytest.mark.skip_xp_backend(Backend.SPARSE, reason="index by sparse array")
    @pytest.mark.skip_xp_backend(Backend.ARRAY_API_STRICTEST, reason="unknown shape")
    def test_none_shape(self, xp: ArrayNamespace):
        a = xp.asarray([1, 5, 0])
        b = xp.asarray([1, 4, 2])
        b = b[a < 5]
        a = a[a < 5]
        assert_equal(isclose(a, b), xp.asarray([True, False]))

    @pytest.mark.skip_xp_backend(Backend.SPARSE, reason="index by sparse array")
    @pytest.mark.skip_xp_backend(Backend.ARRAY_API_STRICTEST, reason="unknown shape")
    def test_none_shape_bool(self, xp: ArrayNamespace):
        a = xp.asarray([True, True, False])
        b = xp.asarray([True, False, True])
        b = b[a]
        a = a[a]
        assert_equal(isclose(a, b), xp.asarray([True, False]))

    @pytest.mark.skip_xp_backend(Backend.NUMPY_READONLY, reason="xp=xp")
    def test_python_scalar(self, xp: ArrayNamespace):
        a = xp.asarray([0.0, 0.1], dtype=xp.float32)
        assert_equal(isclose(a, 0.0), xp.asarray([True, False]))
        assert_equal(isclose(0.0, a), xp.asarray([True, False]))

        a = xp.asarray([0, 1], dtype=xp.int16)
        assert_equal(isclose(a, 0), xp.asarray([True, False]))
        assert_equal(isclose(0, a), xp.asarray([True, False]))

        assert_equal(isclose(0, 0, xp=xp), xp.asarray(True))
        assert_equal(isclose(0, 1, xp=xp), xp.asarray(False))

    def test_all_python_scalars(self):
        with pytest.raises(TypeError, match=r"array_namespace requires .* array input"):
            _ = isclose(0, 0)

    def test_xp(self, xp: ArrayNamespace):
        a = xp.asarray([0.0, 0.0])
        b = xp.asarray([1e-9, 1e-4])
        assert_equal(isclose(a, b, xp=xp), xp.asarray([True, False]))

    @pytest.mark.parametrize("equal_nan", [True, False])
    def test_device(self, xp: ArrayNamespace, device: Device, equal_nan: bool):
        a = xp.asarray([0.0, 0.0, xp.nan], device=device)
        b = xp.asarray([1e-9, 1e-4, xp.nan], device=device)
        res = isclose(a, b, equal_nan=equal_nan)
        assert get_device(res) == device

    def test_array_on_device_with_scalar(self, xp: ArrayNamespace, device: Device):
        a = xp.asarray([0.01, 0.5, 0.8, 0.9, 1.00001], device=device, dtype=xp.float64)
        b = 1
        res = isclose(a, b)
        assert get_device(res) == device
        assert_equal(res, xp.asarray([False, False, False, False, True], device=device))

        a = 0.1
        b = xp.asarray([0.01, 0.5, 0.8, 0.9, 0.100001], device=device, dtype=xp.float64)
        res = isclose(a, b)
        assert get_device(res) == device
        assert_equal(res, xp.asarray([False, False, False, False, True], device=device))


class TestNanToNum:
    def test_bool(self, xp: ArrayNamespace) -> None:
        a = xp.asarray([True])
        assert_equal(nan_to_num(a, xp=xp), a)

    def test_scalar_pos_inf(self, xp: ArrayNamespace, infinity: float) -> None:
        a = xp.inf
        assert_equal(nan_to_num(a, xp=xp), xp.asarray(infinity))

    def test_scalar_neg_inf(self, xp: ArrayNamespace, infinity: float) -> None:
        a = -xp.inf
        assert_equal(nan_to_num(a, xp=xp), -xp.asarray(infinity))

    def test_scalar_nan(self, xp: ArrayNamespace) -> None:
        a = xp.nan
        assert_equal(nan_to_num(a, xp=xp), xp.asarray(0.0))

    def test_real(self, xp: ArrayNamespace, infinity: float) -> None:
        a = xp.asarray([xp.inf, -xp.inf, xp.nan, -128, 128])
        assert_equal(
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

    def test_complex(self, xp: ArrayNamespace, infinity: float) -> None:
        a = xp.asarray(
            [
                complex(xp.inf, xp.nan),
                xp.nan,
                complex(xp.nan, xp.inf),
            ]
        )
        assert_equal(
            nan_to_num(a),
            xp.asarray([complex(infinity, 0), complex(0, 0), complex(0, infinity)]),
        )

    def test_empty_array(self, xp: ArrayNamespace) -> None:
        a = xp.asarray([], dtype=xp.float32)  # forced dtype due to torch
        assert_equal(nan_to_num(a, xp=xp), a)
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
        xp: ArrayNamespace,
        in_vals: Array,
        fill_value: float,
        out_vals: Array,
    ) -> None:
        a = xp.asarray(in_vals)
        assert_equal(
            nan_to_num(a, fill_value=fill_value, xp=xp),
            xp.asarray(out_vals),
        )

    def test_fill_value_failure(self, xp: ArrayNamespace) -> None:
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


class TestSinc:
    def test_simple(self, xp: ArrayNamespace):
        assert_equal(sinc(xp.asarray(0.0)), xp.asarray(1.0))
        x = xp.asarray(np.linspace(-1, 1, 100))
        w = sinc(x)
        # check symmetry
        assert_close(w, xp.flip(w, axis=0))

    @pytest.mark.parametrize("x", [0, 1 + 3j])
    def test_dtype(self, xp: ArrayNamespace, x: complex):
        with pytest.raises(ValueError, match="real floating data type"):
            _ = sinc(xp.asarray(x))

    def test_3d(self, xp: ArrayNamespace):
        x = np.arange(18, dtype=np.float64).reshape((3, 3, 2))
        expected = np.zeros_like(x)
        expected[0, 0, 0] = 1
        x = xp.asarray(x)
        expected = xp.asarray(expected)
        assert_close(sinc(x), expected, atol=1e-15)

    def test_device(self, xp: ArrayNamespace, device: Device):
        x = xp.asarray(0.0, device=device)
        assert get_device(sinc(x)) == device

    def test_xp(self, xp: ArrayNamespace):
        assert_equal(sinc(xp.asarray(0.0), xp=xp), xp.asarray(1.0))


class TestAngle:
    def test_simple(self, xp: ArrayNamespace):
        a = xp.asarray([1, 0])
        res = angle(a)
        expected = xp.asarray([0.0, 0.0], dtype=res.dtype)
        assert_equal(res, expected)

    def test_basic(self, xp: ArrayNamespace):
        x = xp.asarray(
            [
                1 + 3j,
                np.sqrt(2) / 2.0 + 1j * np.sqrt(2) / 2,
                1,
                1j,
                -1,
                -1j,
                1 - 3j,
                -1 + 3j,
            ],
            dtype=xp.complex128,
        )
        expected = xp.asarray(
            [
                np.arctan(3.0 / 1.0),
                np.arctan(1.0),
                0,
                np.pi / 2,
                np.pi,
                -np.pi / 2.0,
                -np.arctan(3.0 / 1.0),
                np.pi - np.arctan(3.0 / 1.0),
            ],
            dtype=xp.float64,
        )
        assert_close(angle(x, xp=xp), expected, rtol=0, atol=1e-11)
        assert_close(
            angle(x, deg=True, xp=xp),
            expected * 180 / xp.pi,
            rtol=0,
            atol=1e-11,
        )

    def test_real(self, xp: ArrayNamespace):
        x = xp.asarray([0.0, -0.0, 1.0, -1.0])
        expected = xp.asarray([0.0, xp.pi, 0.0, xp.pi], dtype=x.dtype)
        assert_close(angle(x, xp=xp), expected)

    def test_complex(self, xp: ArrayNamespace):
        a = xp.asarray([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])
        expected = xp.asarray([xp.pi / 4, -xp.pi / 4, 3 * xp.pi / 4, -3 * xp.pi / 4])
        res = angle(a, xp=xp)
        assert_equal(res, expected)

    def test_integral(self, xp: ArrayNamespace):
        x = xp.asarray([0, -1, 1], dtype=xp.int32)
        actual = angle(x, xp=xp)
        expected = xp.asarray(
            [0.0, xp.pi, 0.0], dtype=default_dtype(xp, device=get_device(x))
        )
        assert_close(actual, expected)

    def test_2d(self, xp: ArrayNamespace):
        a = xp.asarray([[1 + 1j, 1 - 1j], [-1 + 1j, -1 - 1j]])
        expected = xp.asarray(
            [[xp.pi / 4, -xp.pi / 4], [3 * xp.pi / 4, -3 * xp.pi / 4]]
        )
        res = angle(a, xp=xp)
        assert_equal(res, expected)

    @pytest.mark.skip_xp_backend(Backend.TORCH, reason="materialize 'meta' device")
    def test_device(self, xp: ArrayNamespace, device: Device):
        a = xp.asarray([1 + 1j], device=device)
        assert get_device(angle(a)) == device


class TestDeg2Rad:
    def test_basic(self, xp: ArrayNamespace):
        x = xp.asarray([0.0, 90.0, 180.0, 270.0, 360.0])
        expected = xp.asarray([0.0, xp.pi / 2, xp.pi, 3 * xp.pi / 2, 2 * xp.pi])
        assert_close(deg2rad(x), expected)

    @pytest.mark.parametrize("dtype_name", ["int32", "int64"])
    def test_integral(self, xp: ArrayNamespace, dtype_name: str):
        x = xp.asarray([0, 90, 180], dtype=getattr(xp, dtype_name))
        actual = deg2rad(x, xp=xp)
        expected = xp.asarray(
            [0.0, xp.pi / 2, xp.pi], dtype=default_dtype(xp, device=get_device(x))
        )
        assert actual.dtype == expected.dtype
        assert_close(actual, expected)

    def test_complex(self, xp: ArrayNamespace):
        x = xp.asarray([180 + 90j], dtype=xp.complex64)
        actual = deg2rad(x, xp=xp)
        expected = xp.asarray([xp.pi + xp.pi / 2 * 1j], dtype=x.dtype)
        assert actual.dtype == x.dtype
        assert_close(actual, expected)

    def test_bool(self, xp: ArrayNamespace):
        x = xp.asarray([True])
        with pytest.raises(TypeError, match="integral, real floating, or complex"):
            _ = deg2rad(x, xp=xp)

    def test_device(self, xp: ArrayNamespace, device: Device):
        x = xp.asarray([0.0, 90.0, 180.0], device=device)
        assert get_device(deg2rad(x)) == device


class TestRad2Deg:
    def test_basic(self, xp: ArrayNamespace):
        x = xp.asarray([0.0, xp.pi / 2, xp.pi, 3 * xp.pi / 2, 2 * xp.pi])
        expected = xp.asarray([0.0, 90.0, 180.0, 270.0, 360.0])
        assert_close(rad2deg(x), expected)

    @pytest.mark.parametrize("dtype_name", ["int32", "int64"])
    def test_integral(self, xp: ArrayNamespace, dtype_name: str):
        x = xp.asarray([0, 1, 2], dtype=getattr(xp, dtype_name))
        actual = rad2deg(x, xp=xp)
        expected = xp.asarray(
            [0.0, 180 / xp.pi, 360 / xp.pi],
            dtype=default_dtype(xp, device=get_device(x)),
        )
        assert actual.dtype == expected.dtype
        assert_close(actual, expected)

    def test_complex(self, xp: ArrayNamespace):
        x = xp.asarray([xp.pi + xp.pi / 2 * 1j], dtype=xp.complex64)
        actual = rad2deg(x, xp=xp)
        expected = xp.asarray([180 + 90j], dtype=x.dtype)
        assert actual.dtype == x.dtype
        assert_close(actual, expected)

    def test_bool(self, xp: ArrayNamespace):
        x = xp.asarray([True])
        with pytest.raises(TypeError, match="integral, real floating, or complex"):
            _ = rad2deg(x, xp=xp)

    def test_device(self, xp: ArrayNamespace, device: Device):
        x = xp.asarray([0.0, xp.pi / 2, xp.pi], device=device)
        assert get_device(rad2deg(x)) == device
