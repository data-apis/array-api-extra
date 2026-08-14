import pytest

from array_api_extra import _agnostic, isin, nunique, setdiff1d, union1d
from array_api_extra._lib._backends import NUMPY_VERSION, Backend
from array_api_extra._lib._compat import device as get_device
from array_api_extra._lib._typing import Array, ArrayNamespace, Device
from array_api_extra.testing import assert_equal, lazy_xp_function

lazy_xp_function(isin)
lazy_xp_function(nunique)
# FIXME calls in1d which calls xp.unique_values without size
lazy_xp_function(setdiff1d, jax_jit=False)
lazy_xp_function(union1d, jax_jit=False)


class TestNUnique:
    @pytest.mark.skip_xp_backend(
        Backend.ARRAY_API_STRICT, reason="array-agnostic fallback"
    )
    @pytest.mark.skip_xp_backend(
        Backend.ARRAY_API_STRICTEST, reason="array-agnostic fallback"
    )
    @pytest.mark.skip_xp_backend(Backend.DASK, reason="array-agnostic fallback")
    @pytest.mark.skip_xp_backend(Backend.SPARSE, reason="array-agnostic fallback")
    def test_delegates(
        self,
        xp: ArrayNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ):
        def fallback(*_args: object, **_kwargs: object) -> Array:
            msg = "array-agnostic fallback should not be used"
            raise AssertionError(msg)

        monkeypatch.setattr(_agnostic._set, "nunique", fallback)
        a = xp.asarray([1, 1, 2])
        assert_equal(nunique(a), xp.asarray(2))

    def test_simple(self, xp: ArrayNamespace):
        a = xp.asarray([[1, 1], [0, 2], [2, 2]])
        assert_equal(nunique(a), xp.asarray(3))

    def test_empty(self, xp: ArrayNamespace):
        a = xp.asarray([])
        assert_equal(nunique(a), xp.asarray(0))

    def test_size1(self, xp: ArrayNamespace):
        a = xp.asarray([123])
        assert_equal(nunique(a), xp.asarray(1))

    def test_all_equal(self, xp: ArrayNamespace):
        a = xp.asarray([123, 123, 123])
        assert_equal(nunique(a), xp.asarray(1))

    @pytest.mark.xfail_xp_backend(Backend.DASK, reason="No equal_nan kwarg in unique")
    def test_nan(self, xp: ArrayNamespace, library: Backend):
        if library.like(Backend.NUMPY) and NUMPY_VERSION < (1, 24):
            pytest.xfail("NumPy <1.24 has no equal_nan kwarg in unique")

        # Each NaN is counted separately
        a = xp.asarray([xp.nan, 123.0, xp.nan])
        assert_equal(nunique(a), xp.asarray(3))

    @pytest.mark.parametrize("size", [0, 1, 2])
    def test_device(self, xp: ArrayNamespace, device: Device, size: int):
        a = xp.asarray([0.0] * size, device=device)
        assert get_device(nunique(a)) == device

    def test_xp(self, xp: ArrayNamespace):
        a = xp.asarray([[1, 1], [0, 2], [2, 2]])
        assert_equal(nunique(a, xp=xp), xp.asarray(3))


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
    def test_setdiff1d(self, xp: ArrayNamespace):
        x1 = xp.asarray([6, 5, 4, 7, 1, 2, 7, 4])
        x2 = xp.asarray([2, 4, 3, 3, 2, 1, 5])

        expected = xp.asarray([6, 7])
        actual = setdiff1d(x1, x2)
        assert_equal(actual, expected)

        x1 = xp.arange(21)
        x2 = xp.arange(19)
        expected = xp.asarray([19, 20])
        actual = setdiff1d(x1, x2)
        assert_equal(actual, expected)

        assert_equal(setdiff1d(xp.empty(0), xp.empty(0)), xp.empty(0))
        x1 = xp.empty(0, dtype=xp.uint32)
        x2 = x1
        assert xp.isdtype(setdiff1d(x1, x2).dtype, xp.uint32)

    def test_assume_unique(self, xp: ArrayNamespace):
        x1 = xp.asarray([3, 2, 1])
        x2 = xp.asarray([7, 5, 2])
        expected = xp.asarray([3, 1])
        actual = setdiff1d(x1, x2, assume_unique=True)
        assert_equal(actual, expected)

    @assume_unique
    @pytest.mark.parametrize("shape1", [(), (1,), (1, 1)])
    @pytest.mark.parametrize("shape2", [(), (1,), (1, 1)])
    def test_shapes(
        self,
        assume_unique: bool,
        shape1: tuple[int, ...],
        shape2: tuple[int, ...],
        xp: ArrayNamespace,
    ):
        x1 = xp.zeros(shape1)
        x2 = xp.zeros(shape2)

        actual = setdiff1d(x1, x2, assume_unique=assume_unique)
        assert_equal(actual, xp.empty((0,)))

    @assume_unique
    @pytest.mark.skip_xp_backend(Backend.NUMPY_READONLY, reason="xp=xp")
    def test_python_scalar(self, xp: ArrayNamespace, assume_unique: bool):
        # Test no dtype promotion to xp.asarray(x2); use x1.dtype
        x1 = xp.asarray([3, 1, 2], dtype=xp.int16)
        x2 = 3
        actual = setdiff1d(x1, x2, assume_unique=assume_unique)
        assert_equal(actual, xp.asarray([1, 2], dtype=xp.int16))

        actual = setdiff1d(x2, x1, assume_unique=assume_unique)
        assert_equal(actual, xp.asarray([], dtype=xp.int16))

        assert_equal(
            setdiff1d(0, 0, assume_unique=assume_unique, xp=xp),
            xp.asarray([0])[:0],  # Default int dtype for backend
        )

    @pytest.mark.parametrize("assume_unique", [True, False])
    def test_all_python_scalars(self, assume_unique: bool):
        with pytest.raises(TypeError, match=r"array_namespace requires .* array input"):
            _ = setdiff1d(0, 0, assume_unique=assume_unique)

    @assume_unique
    @pytest.mark.skip_xp_backend(
        Backend.TORCH, reason="device='meta' does not support unknown shapes"
    )
    def test_device(self, xp: ArrayNamespace, device: Device, assume_unique: bool):
        x1 = xp.asarray([3, 8, 20], device=device)
        x2 = xp.asarray([2, 3, 4], device=device)
        assert get_device(setdiff1d(x1, x2, assume_unique=assume_unique)) == device

    @pytest.mark.skip_xp_backend(Backend.NUMPY_READONLY, reason="xp=xp")
    def test_xp(self, xp: ArrayNamespace):
        x1 = xp.asarray([3, 8, 20])
        x2 = xp.asarray([2, 3, 4])
        expected = xp.asarray([8, 20])
        actual = setdiff1d(x1, x2, assume_unique=True, xp=xp)
        assert_equal(actual, expected)


@pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no unique_inverse")
class TestIsIn:
    def test_simple(self, xp: ArrayNamespace, library: Backend):
        if library.like(Backend.NUMPY) and NUMPY_VERSION < (1, 24):
            pytest.xfail("NumPy <1.24 has no kind kwarg in isin")

        b = xp.asarray([1, 2, 3, 4])

        # `a` with 1 dimension
        a = xp.asarray([1, 3, 6, 10])
        expected = xp.asarray([True, True, False, False])
        res = isin(a, b)
        assert_equal(res, expected)

        # `a` with 2 dimensions
        a = xp.asarray([[0, 2], [4, 6]])
        expected = xp.asarray([[False, True], [True, False]])
        res = isin(a, b)
        assert_equal(res, expected)

    def test_device(self, xp: ArrayNamespace, device: Device, library: Backend):
        if library.like(Backend.NUMPY) and NUMPY_VERSION < (1, 24):
            pytest.xfail("NumPy <1.24 has no kind kwarg in isin")

        a = xp.asarray([1, 3, 6], device=device)
        b = xp.asarray([1, 2, 3], device=device)
        assert get_device(isin(a, b)) == device

    def test_assume_unique_and_invert(
        self, xp: ArrayNamespace, device: Device, library: Backend
    ):
        if library.like(Backend.NUMPY) and NUMPY_VERSION < (1, 24):
            pytest.xfail("NumPy <1.24 has no kind kwarg in isin")

        a = xp.asarray([0, 3, 6, 10], device=device)
        b = xp.asarray([1, 2, 3, 10], device=device)
        expected = xp.asarray([True, False, True, False], device=device)
        res = isin(a, b, assume_unique=True, invert=True)
        assert get_device(res) == device
        assert_equal(res, expected)

    def test_kind(self, xp: ArrayNamespace, library: Backend):
        if library.like(Backend.NUMPY) and NUMPY_VERSION < (1, 24):
            pytest.xfail("NumPy <1.24 has no kind kwarg in isin")

        a = xp.asarray([0, 3, 6, 10])
        b = xp.asarray([1, 2, 3, 10])
        expected = xp.asarray([False, True, False, True])
        res = isin(a, b, kind="sort")
        assert_equal(res, expected)


@pytest.mark.skip_xp_backend(
    Backend.ARRAY_API_STRICTEST,
    reason="data_dependent_shapes flag for unique_values is disabled",
)
class TestUnion1d:
    def test_simple(self, xp: ArrayNamespace):
        a = xp.asarray([-1, 1, 0])
        b = xp.asarray([2, -2, 0])
        expected = xp.asarray([-2, -1, 0, 1, 2])
        res = union1d(a, b)
        assert_equal(res, expected)

    def test_2d(self, xp: ArrayNamespace):
        a = xp.asarray([[-1, 1, 0], [1, 2, 0]])
        b = xp.asarray([[1, 0, 1], [-2, -1, 0]])
        expected = xp.asarray([-2, -1, 0, 1, 2])
        res = union1d(a, b)
        assert_equal(res, expected)

    def test_3d(self, xp: ArrayNamespace):
        a = xp.asarray([[[-1, 0], [1, 2]], [[-1, 0], [1, 2]]])
        b = xp.asarray([[[0, 1], [-1, 2]], [[1, -2], [0, 2]]])
        expected = xp.asarray([-2, -1, 0, 1, 2])
        res = union1d(a, b)
        assert_equal(res, expected)

    @pytest.mark.skip_xp_backend(Backend.TORCH, reason="materialize 'meta' device")
    def test_device(self, xp: ArrayNamespace, device: Device):
        a = xp.asarray([-1, 1, 0], device=device)
        b = xp.asarray([2, -2, 0], device=device)
        assert get_device(union1d(a, b)) == device
