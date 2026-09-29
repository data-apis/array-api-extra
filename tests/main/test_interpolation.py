from typing import Any, Literal

import numpy as np
import pytest

from array_api_extra import interp as xpx_interp
from array_api_extra._agnostic._interpolation import interp as agnostic_interp
from array_api_extra._lib._backends import Backend
from array_api_extra._lib._compat import array_namespace
from array_api_extra._lib._compat import device as get_device
from array_api_extra._lib._compat import is_array_api_obj
from array_api_extra._lib._typing import Array, ArrayNamespace, Device
from array_api_extra.testing import assert_close, assert_equal


Implementation = Literal["public", "agnostic"]

implementations = pytest.mark.parametrize(
    "implementation", ["public", "agnostic"]
)


def _interp(
    implementation: Implementation,
    x: Any,
    x_points: Array,
    values: Array,
    /,
    *,
    xp: ArrayNamespace,
    left: Any = None,
    right: Any = None,
    period: Any = None,
) -> Array:
    if implementation == "public":
        return xpx_interp(
            x, x_points, values, left=left, right=right, period=period
        )

    coordinate_device = get_device(x_points)
    if is_array_api_obj(x):
        x = xp.astype(x, xp.float64, copy=False)
    else:
        x = xp.asarray(x, dtype=xp.float64, device=coordinate_device)
    x_points = xp.astype(x_points, xp.float64, copy=False)

    value_dtype = (
        xp.complex128
        if xp.isdtype(values.dtype, "complex floating")
        else xp.float64
    )
    values = xp.astype(values, value_dtype, copy=False)
    if period is not None:
        period = abs(period)
        left = right = None
    else:
        if left is not None:
            left = (
                xp.astype(left, value_dtype, copy=False)
                if is_array_api_obj(left)
                else xp.asarray(left, dtype=value_dtype, device=get_device(values))
            )
        if right is not None:
            right = (
                xp.astype(right, value_dtype, copy=False)
                if is_array_api_obj(right)
                else xp.asarray(right, dtype=value_dtype, device=get_device(values))
            )
    return agnostic_interp(
        x,
        x_points,
        values,
        left=left,
        right=right,
        period=period,
        xp=xp,
    )


@pytest.mark.skip_xp_backend(Backend.SPARSE, reason="no searchsorted")
@pytest.mark.skip_xp_backend(Backend.MPARRAY, reason="no searchsorted")
class TestInterp:
    @implementations
    def test_finite_values_and_shapes(
        self, xp: ArrayNamespace, implementation: Implementation
    ):
        x_points = xp.asarray([0, 1, 2], dtype=xp.float32)
        values = xp.asarray([0, 10, 20], dtype=xp.float32)

        actual = _interp(
            implementation,
            xp.asarray([[-1, 0.5], [1.5, 3]], dtype=xp.float32),
            x_points,
            values,
            xp=xp,
        )
        expected = xp.asarray([[0, 5], [15, 20]], dtype=xp.float64)
        assert_close(actual, expected)
        assert array_namespace(actual) == array_namespace(x_points)

        array_scalar = xp.asarray(0.5, dtype=xp.float64)
        queries = (
            (0.5, array_scalar)
            if implementation == "public"
            else (array_scalar,)
        )
        for x in queries:
            actual = _interp(implementation, x, x_points, values, xp=xp)
            assert actual.shape == ()
            assert_equal(actual, xp.asarray(5, dtype=xp.float64))

        actual = _interp(
            implementation,
            xp.asarray([], dtype=xp.float64),
            x_points,
            values,
            xp=xp,
        )
        assert_equal(actual, xp.asarray([], dtype=xp.float64))

    @pytest.mark.parametrize(
        "values",
        [[0.0, 1.0], [0.0 + 0.0j, 1.0 + 2.0j]],
        ids=["real", "complex"],
    )
    @implementations
    def test_numpy_oracle_narrow_interval(
        self,
        xp: ArrayNamespace,
        implementation: Implementation,
        values: list[float] | list[complex],
    ):
        x_points = np.asarray([0.0, 1e-20])
        x = np.asarray([0.0, 2.5e-21, 1e-20])
        expected = np.interp(x, x_points, values)
        dtype = xp.complex128 if np.iscomplexobj(expected) else xp.float64

        actual = _interp(
            implementation,
            xp.asarray(x, dtype=xp.float64),
            xp.asarray(x_points, dtype=xp.float64),
            xp.asarray(values, dtype=dtype),
            xp=xp,
        )
        assert_close(actual, xp.asarray(expected, dtype=dtype))

    @implementations
    def test_numpy_oracle_extreme_finite_coordinates(
        self, xp: ArrayNamespace, implementation: Implementation
    ):
        x_points = np.asarray([-1e308, 1e308])
        x = np.asarray([5e307])
        values = np.asarray([0.0, 1.0])
        with np.errstate(over="ignore", invalid="ignore"):
            expected = np.interp(x, x_points, values)

        actual = _interp(
            implementation,
            xp.asarray(x, dtype=xp.float64),
            xp.asarray(x_points, dtype=xp.float64),
            xp.asarray(values, dtype=xp.float64),
            xp=xp,
        )
        assert_equal(actual, xp.asarray(expected, dtype=xp.float64))

    # NumPy recommends strictly increasing sample coordinates. array-api-extra
    # deliberately defines these repeated-knot cases, including which value wins at
    # the exact knot, rather than exposing searchsorted clipping or a 0/0 division.
    @pytest.mark.parametrize(
        ("x_points", "values", "x", "expected"),
        [
            ([0, 0, 1], [1, 2, 4], [-0.5, 0, 0.5], [1, 2, 3]),
            ([0, 1, 1, 2], [0, 10, 20, 30], [0.5, 1, 1.5], [5, 20, 25]),
            ([0, 1, 2, 2], [0, 10, 20, 30], [1.5, 2, 2.5], [15, 30, 30]),
        ],
        ids=["leading", "middle", "trailing"],
    )
    @implementations
    def test_repeated_coordinates(
        self,
        xp: ArrayNamespace,
        implementation: Implementation,
        x_points: list[int],
        values: list[int],
        x: list[float],
        expected: list[float],
    ):
        actual = _interp(
            implementation,
            xp.asarray(x, dtype=xp.float64),
            xp.asarray(x_points, dtype=xp.float64),
            xp.asarray(values, dtype=xp.float64),
            xp=xp,
        )
        assert_close(actual, xp.asarray(expected, dtype=xp.float64))

    @implementations
    def test_singleton_and_nan_query(
        self, xp: ArrayNamespace, implementation: Implementation
    ):
        x = xp.asarray([xp.nan, -3, 99], dtype=xp.float64)
        actual = _interp(
            implementation,
            x,
            xp.asarray([1], dtype=xp.float64),
            xp.asarray([7], dtype=xp.float64),
            xp=xp,
        )
        assert_equal(actual, xp.asarray([7, 7, 7], dtype=xp.float64))

        actual = _interp(
            implementation,
            xp.asarray([xp.nan], dtype=xp.float64),
            xp.asarray([0, 1], dtype=xp.float64),
            xp.asarray([1, 2], dtype=xp.float64),
            xp=xp,
        )
        assert_equal(actual, xp.asarray([xp.nan], dtype=xp.float64))

    @implementations
    def test_left_right_and_complex_values(
        self, xp: ArrayNamespace, implementation: Implementation
    ):
        x_points = xp.asarray([0, 1, 2], dtype=xp.float64)
        values = xp.asarray([0, 2 + 4j, 4], dtype=xp.complex128)
        actual = _interp(
            implementation,
            xp.asarray([-1, 0.5, 1.5, 3], dtype=xp.float64),
            x_points,
            values,
            left=xp.asarray(1 - 1j, dtype=xp.complex128),
            right=5 + 2j,
            xp=xp,
        )
        expected = xp.asarray(
            [1 - 1j, 1 + 2j, 3 + 2j, 5 + 2j], dtype=xp.complex128
        )
        assert_close(actual, expected)

    @implementations
    def test_periodic_unsorted_negative_period_and_ignored_fills(
        self, xp: ArrayNamespace, implementation: Implementation
    ):
        x = xp.asarray([-1, 0, 1, 2, 3, 4, 5], dtype=xp.float64)
        x_points = xp.asarray([3, 1], dtype=xp.float64)
        values = xp.asarray([30, 10], dtype=xp.float64)

        actual = _interp(
            implementation,
            x,
            x_points,
            values,
            left="ignored",
            right=False,
            period=-4.0,
            xp=xp,
        )
        expected = xp.asarray([30, 20, 10, 20, 30, 20, 10], dtype=xp.float64)
        assert_close(actual, expected)

        assert_equal(x, xp.asarray([-1, 0, 1, 2, 3, 4, 5], dtype=xp.float64))
        assert_equal(x_points, xp.asarray([3, 1], dtype=xp.float64))
        assert_equal(values, xp.asarray([30, 10], dtype=xp.float64))

    @implementations
    def test_integer_inputs_and_large_coordinates(
        self, xp: ArrayNamespace, implementation: Implementation
    ):
        actual = _interp(
            implementation,
            xp.asarray([0, 1, 2], dtype=xp.int64),
            xp.asarray([0, 2], dtype=xp.int64),
            xp.asarray([1, 5], dtype=xp.int64),
            xp=xp,
        )
        assert_close(actual, xp.asarray([1, 3, 5], dtype=xp.float64))

        base = 2**24
        actual = _interp(
            implementation,
            xp.asarray(
                [base, base + 0.5, base + 1, base + 1.5, base + 2],
                dtype=xp.float64,
            ),
            xp.asarray([base, base + 1, base + 2], dtype=xp.int64),
            xp.asarray([0, 10, 20], dtype=xp.int64),
            xp=xp,
        )
        assert_close(actual, xp.asarray([0, 5, 10, 15, 20], dtype=xp.float64))

    @implementations
    def test_nonfinite_values(
        self, xp: ArrayNamespace, implementation: Implementation
    ):
        x_points = xp.asarray([0, 1, 2], dtype=xp.float64)
        actual = _interp(
            implementation,
            x_points,
            x_points,
            xp.asarray([-xp.inf, 2, xp.inf], dtype=xp.float64),
            xp=xp,
        )
        assert_equal(actual, xp.asarray([-xp.inf, 2, xp.inf], dtype=xp.float64))

        actual = _interp(
            implementation,
            xp.asarray([0.5], dtype=xp.float64),
            xp.asarray([0, 1], dtype=xp.float64),
            xp.asarray([xp.inf, xp.inf], dtype=xp.float64),
            xp=xp,
        )
        assert_equal(actual, xp.asarray([xp.inf], dtype=xp.float64))

        actual = _interp(
            implementation,
            xp.asarray([0, 0.5, 1], dtype=xp.float64),
            xp.asarray([0, 1], dtype=xp.float64),
            xp.asarray([np.nan, 2], dtype=xp.float64),
            xp=xp,
        )
        assert_equal(actual, xp.asarray([xp.nan, xp.nan, 2], dtype=xp.float64))

        actual = _interp(
            implementation,
            xp.asarray([0.5], dtype=xp.float64),
            xp.asarray([0, 1], dtype=xp.float64),
            xp.asarray(
                [complex(np.inf, 0), complex(np.inf, 2)], dtype=xp.complex128
            ),
            xp=xp,
        )
        assert_equal(
            actual, xp.asarray([complex(np.inf, 1)], dtype=xp.complex128)
        )

    @implementations
    @pytest.mark.skip_xp_backend(
        Backend.TORCH, reason="device='meta' does not support searchsorted"
    )
    def test_device(
        self,
        xp: ArrayNamespace,
        device: Device,
        implementation: Implementation,
    ):
        actual = _interp(
            implementation,
            xp.asarray([0.5], dtype=xp.float64, device=device),
            xp.asarray([0, 1], dtype=xp.float64, device=device),
            xp.asarray([0, 2], dtype=xp.float64, device=device),
            xp=xp,
        )
        assert get_device(actual) == device

    @pytest.mark.skip_xp_backend(Backend.NUMPY_READONLY, reason="xp=xp")
    def test_xp_keyword(self, xp: ArrayNamespace):
        actual = xpx_interp(
            xp.asarray([0.5], dtype=xp.float64),
            xp.asarray([0, 1], dtype=xp.float64),
            xp.asarray([0, 2], dtype=xp.float64),
            xp=xp,
        )
        assert_equal(actual, xp.asarray([1], dtype=xp.float64))

    def test_shape_and_empty_validation(self, xp: ArrayNamespace):
        x = xp.asarray([0.5], dtype=xp.float64)
        x_points = xp.asarray([0, 1], dtype=xp.float64)
        values = xp.asarray([0, 1], dtype=xp.float64)

        with pytest.raises(ValueError):
            xpx_interp(x, xp.reshape(x_points, (1, 2)), values)
        with pytest.raises(ValueError):
            xpx_interp(x, x_points, xp.reshape(values, (1, 2)))
        with pytest.raises(ValueError):
            xpx_interp(x, x_points, values[:1])
        with pytest.raises(ValueError):
            xpx_interp(x, x_points[:0], values[:0])

    def test_type_validation(self, xp: ArrayNamespace):
        x = xp.asarray([0.5], dtype=xp.float64)
        x_points = xp.asarray([0, 1], dtype=xp.float64)
        values = xp.asarray([0, 1], dtype=xp.float64)

        with pytest.raises(TypeError):
            xpx_interp([0.5], x_points, values)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            xpx_interp(x, [0, 1], values)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            xpx_interp(x, x_points, [0, 1])  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            xpx_interp(xp.asarray([True]), x_points, values)
        with pytest.raises(TypeError):
            xpx_interp(x, xp.asarray([False, True]), values)
        with pytest.raises(TypeError):
            xpx_interp(x, x_points, xp.asarray([False, True]))
        with pytest.raises(TypeError):
            xpx_interp(True, x_points, values)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            xpx_interp(1j, x_points, values)  # type: ignore[arg-type]

    def test_bound_and_period_validation(self, xp: ArrayNamespace):
        x = xp.asarray([0.5], dtype=xp.float64)
        x_points = xp.asarray([0, 1], dtype=xp.float64)
        values = xp.asarray([0, 1], dtype=xp.float64)

        with pytest.raises(ValueError):
            xpx_interp(x, x_points, values, left=xp.asarray([0]))
        with pytest.raises(TypeError):
            xpx_interp(x, x_points, values, right="bad")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            xpx_interp(x, x_points, values, left=1j)

        for period in (0.0, np.inf, -np.inf, np.nan):
            with pytest.raises(ValueError):
                xpx_interp(x, x_points, values, period=period)
        with pytest.raises(TypeError):
            xpx_interp(x, x_points, values, period=True)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            xpx_interp(x, x_points, values, period=xp.asarray(4.0))  # type: ignore[arg-type]
