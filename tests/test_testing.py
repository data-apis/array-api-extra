import numpy as np
import pytest

from array_api_extra._lib import Library
from array_api_extra._lib._testing import xp_assert_close, xp_assert_equal

# mypy: disable-error-code=no-any-decorated
# pyright: reportUnknownParameterType=false,reportMissingParameterType=false


@pytest.mark.parametrize(
    "func",
    [
        xp_assert_equal,
        pytest.param(
            xp_assert_close,
            marks=pytest.mark.skip_xp_backend(Library.SPARSE, reason="no isdtype"),
        ),
    ],
)
def test_assert_close_equal_basic(xp, func):
    func(xp.asarray(0), xp.asarray(0))
    func(xp.asarray([1, 2]), xp.asarray([1, 2]))

    with pytest.raises(AssertionError, match="shapes do not match"):
        func(xp.asarray([0]), xp.asarray([[0]]))

    with pytest.raises(AssertionError, match="dtypes do not match"):
        func(xp.asarray(0, dtype=xp.float32), xp.asarray(0, dtype=xp.float64))

    with pytest.raises(AssertionError):
        func(xp.asarray([1, 2]), xp.asarray([1, 3]))

    with pytest.raises(AssertionError, match="hello"):
        func(xp.asarray([1, 2]), xp.asarray([1, 3]), err_msg="hello")


@pytest.mark.skip_xp_backend(Library.NUMPY)
@pytest.mark.skip_xp_backend(Library.NUMPY_READONLY)
@pytest.mark.parametrize(
    "func",
    [
        xp_assert_equal,
        pytest.param(
            xp_assert_close,
            marks=pytest.mark.skip_xp_backend(Library.SPARSE, reason="no isdtype"),
        ),
    ],
)
def test_assert_close_equal_namespace(xp, func):
    with pytest.raises(AssertionError):
        func(xp.asarray(0), np.asarray(0))
    with pytest.raises(TypeError):
        func(xp.asarray(0), 0)
    with pytest.raises(TypeError):
        func(xp.asarray([0]), [0])


@pytest.mark.skip_xp_backend(Library.SPARSE, reason="no isdtype")
def test_assert_close_tolerance(xp):
    xp_assert_close(xp.asarray([100.0]), xp.asarray([102.0]), rtol=0.03)
    with pytest.raises(AssertionError):
        xp_assert_close(xp.asarray([100.0]), xp.asarray([102.0]), rtol=0.01)

    xp_assert_close(xp.asarray([100.0]), xp.asarray([102.0]), atol=3)
    with pytest.raises(AssertionError):
        xp_assert_close(xp.asarray([100.0]), xp.asarray([102.0]), atol=1)
