from types import ModuleType
from typing import TYPE_CHECKING, Generic, TypeVar, cast

import numpy as np
import pytest

from array_api_extra._lib._backends import Backend
from array_api_extra._lib._testing import xp_assert_equal
from array_api_extra._lib._utils._compat import array_namespace
from array_api_extra._lib._utils._compat import device as get_device
from array_api_extra._lib._utils._helpers import (
    asarrays,
    capabilities,
    eager_shape,
    in1d,
    jax_autojit,
    meta_namespace,
    ndindex,
    pickle_flatten,
    pickle_unflatten,
)
from array_api_extra._lib._utils._typing import Array, Device, DType
from array_api_extra.testing import lazy_xp_function

from .conftest import np_compat

if TYPE_CHECKING:
    # TODO import from typing (requires Python >=3.12)
    from typing_extensions import override
else:

    def override(func):
        return func

# mypy: disable-error-code=no-untyped-usage

T = TypeVar("T")

# FIXME calls xp.unique_values without size
lazy_xp_function(in1d, jax_jit=False)


@pytest.mark.skip_xp_backend(Backend.SPARSE, reason="no unique_inverse")
@pytest.mark.skip_xp_backend(Backend.ARRAY_API_STRICTEST, reason="no unique_inverse")
class TestIn1D:
    # cover both code paths
    @pytest.mark.parametrize(
        "n",
        [
            pytest.param(9, id="fast path"),
            pytest.param(
                15,
                id="slow path",
                marks=pytest.mark.xfail_xp_backend(
                    Backend.DASK, reason="NaN-shaped array"
                ),
            ),
        ],
    )
    def test_no_invert_assume_unique(self, xp: ModuleType, n: int):
        x1 = xp.asarray([3, 8, 20])
        x2 = xp.arange(n)
        expected = xp.asarray([True, True, False])
        actual = in1d(x1, x2)
        xp_assert_equal(actual, expected)

    def test_device(self, xp: ModuleType, device: Device):
        x1 = xp.asarray([3, 8, 20], device=device)
        x2 = xp.asarray([2, 3, 4], device=device)
        assert get_device(in1d(x1, x2)) == device

    @pytest.mark.skip_xp_backend(Backend.NUMPY_READONLY, reason="xp=xp")
    def test_xp(self, xp: ModuleType):
        x1 = xp.asarray([1, 6])
        x2 = xp.asarray([0, 1, 2, 3, 4])
        expected = xp.asarray([True, False])
        actual = in1d(x1, x2, xp=xp)
        xp_assert_equal(actual, expected)


class TestAsArrays:
    @pytest.mark.xfail_xp_backend(Backend.SPARSE, reason="no isdtype")
    @pytest.mark.parametrize(
        ("dtype", "b", "defined"),
        [
            # Well-defined cases of dtype promotion from Python scalar to Array
            # bool vs. bool
            ("bool", True, True),
            # int vs. xp.*int*, xp.float*, xp.complex*
            ("int16", 1, True),
            ("uint8", 1, True),
            ("float32", 1, True),
            ("float64", 1, True),
            ("complex64", 1, True),
            ("complex128", 1, True),
            # float vs. xp.float, xp.complex
            ("float32", 1.0, True),
            ("float64", 1.0, True),
            ("complex64", 1.0, True),
            ("complex128", 1.0, True),
            # complex vs. xp.complex
            ("complex64", 1.0j, True),
            ("complex128", 1.0j, True),
            # Undefined cases
            ("bool", 1, False),
            ("int64", 1.0, False),
            ("float64", 1.0j, False),
        ],
    )
    def test_array_vs_scalar(
        self, dtype: str, b: complex, defined: bool, xp: ModuleType
    ):
        a = xp.asarray(1, dtype=getattr(xp, dtype))

        xa, xb = asarrays(a, b, xp)
        assert xa.dtype == a.dtype
        if defined:
            assert xb.dtype == a.dtype
        else:
            assert xb.dtype == xp.asarray(b).dtype

        xbr, xar = asarrays(b, a, xp)
        assert xar.dtype == xa.dtype
        assert xbr.dtype == xb.dtype

    def test_scalar_vs_scalar(self, xp: ModuleType):
        a, b = asarrays(1, 2.2, xp=xp)
        assert a.dtype == xp.asarray(1).dtype  # Default dtype
        assert b.dtype == xp.asarray(2.2).dtype  # Default dtype; not broadcasted

    ALL_TYPES: tuple[str, ...] = (
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "bool",
    )

    @pytest.mark.parametrize("a_type", ALL_TYPES)
    @pytest.mark.parametrize("b_type", ALL_TYPES)
    def test_array_vs_array(self, a_type: str, b_type: str, xp: ModuleType):
        """
        Test that when both inputs of asarray are already Array API objects,
        they are returned unchanged.
        """
        a = xp.asarray(1, dtype=getattr(xp, a_type))
        b = xp.asarray(1, dtype=getattr(xp, b_type))
        xa, xb = asarrays(a, b, xp)
        assert xa.dtype == a.dtype
        assert xb.dtype == b.dtype

    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    def test_numpy_generics(self, dtype: DType):
        """
        Test special case of np.float64 and np.complex128,
        which are subclasses of float and complex.
        """
        a = cast(Array, dtype(0))  # type: ignore[operator]  # pyright: ignore[reportCallIssue]
        xa, xb = asarrays(a, 0, xp=np_compat)
        assert xa.dtype == dtype
        assert xb.dtype == dtype


@pytest.mark.parametrize(
    "shape", [(), (1,), (5,), (2, 3), (5, 3, 8), (0,), (3, 0), (0, 0, 1)]
)
def test_ndindex(shape: tuple[int, ...]):
    assert tuple(ndindex(*shape)) == tuple(np.ndindex(*shape))


@pytest.mark.skip_xp_backend(Backend.SPARSE, reason="index by sparse array")
@pytest.mark.skip_xp_backend(Backend.ARRAY_API_STRICTEST, reason="boolean indexing")
def test_eager_shape(xp: ModuleType, library: Backend):
    a = xp.asarray([1, 2, 3])
    # Lazy arrays, like Dask, have an eager shape until you slice them with
    # a lazy boolean mask
    assert eager_shape(a) == a.shape == (3,)

    b = a[a > 2]
    if library is Backend.DASK:
        with pytest.raises(TypeError, match="Unsupported lazy shape"):
            _ = eager_shape(b)
    # FIXME can't test use case for None in the shape until we add support for
    # other lazy backends
    else:
        assert eager_shape(b) == b.shape == (1,)


class TestMetaNamespace:
    @pytest.mark.skip_xp_backend(Backend.NUMPY_READONLY, reason="namespace tests")
    def test_basic(self, xp: ModuleType, library: Backend):
        args = None, xp.asarray(0), 1
        expect = np_compat if library is Backend.DASK else xp
        assert meta_namespace(*args) is expect

    def test_dask_metas(self, da: ModuleType):
        cp = pytest.importorskip("cupy")
        cp_compat = array_namespace(cp.empty(0))
        args = None, da.from_array(cp.asarray(0)), 1
        assert meta_namespace(*args) is cp_compat

    def test_xp(self, xp: ModuleType):
        args = None, xp.asarray(0), 1
        assert meta_namespace(*args, xp=xp) in (xp, np_compat)


def test_capabilities(xp: ModuleType):
    expect = {"boolean indexing", "data-dependent shapes"}
    if xp.__array_api_version__ >= "2024.12":
        expect.add("max dimensions")
    assert capabilities(xp).keys() == expect


class Wrapper(Generic[T]):
    """Trivial opaque wrapper. Must be pickleable."""

    x: T
    # __slots__ make this object serializable with __reduce_ex__(5),
    # but not with __reduce__
    __slots__: tuple[str, ...] = ("x",)

    def __init__(self, x: T):
        self.x = x

    # Note: this makes the object not hashable
    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Wrapper) and self.x == other.x


class TestPickleFlatten:
    def test_roundtrip(self):
        class NotSerializable:
            @override
            def __reduce__(self) -> tuple[object, ...]:
                raise NotImplementedError()

        # Note: NotHashable() instances can be reduced to an
        # unserializable local class
        class NotHashable:
            @override
            def __eq__(self, other: object) -> bool:
                return isinstance(other, type(self)) and other.__dict__ == self.__dict__

        with pytest.raises(TypeError):
            _ = hash(NotHashable())

        # Extracted objects need be neither pickleable nor serializable
        class C(NotSerializable, NotHashable):
            x: int

            def __init__(self, x: int):
                self.x = x

        class D(C):
            pass

        c1 = C(1)
        c2 = C(2)
        d3 = D(3)

        # An assorted bunch of opaque containers, standard containers,
        # non-serializable objects, and non-hashable objects (but not at the same time)
        obj = Wrapper([1, c1, {2: (c2, {NotSerializable()})}, NotHashable(), d3])
        instances, rest = pickle_flatten(obj, C)

        assert instances == [c1, c2, d3]
        obj2 = pickle_unflatten(instances, rest)
        assert obj2 == obj

    def test_swap_objects(self):
        class C:
            pass

        obj = [1, C(), {2: (C(), {C()})}]
        _, rest = pickle_flatten(obj, C)
        obj2 = pickle_unflatten(["foo", "bar", "baz"], rest)
        assert obj2 == [1, "foo", {2: ("bar", {"baz"})}]

    def test_multi_class(self):
        class C:
            pass

        class D:
            pass

        c, d = C(), D()
        instances, _ = pickle_flatten([c, d], (C, D))
        assert len(instances) == 2
        assert instances[0] is c
        assert instances[1] is d

    def test_no_class(self):
        obj = {1: "foo", 2: (3, 4)}
        instances, rest = pickle_flatten(obj, ())  # type: ignore[var-annotated]
        assert instances == []
        obj2 = pickle_unflatten([], rest)
        assert obj2 == obj

    def test_flattened_stream(self):
        """
        Test that multiple calls to flatten() can feed into the same stream of instances
        """
        obj1 = Wrapper(1)
        obj2 = [Wrapper(2), Wrapper(3)]
        instances1, rest1 = pickle_flatten(obj1, Wrapper)
        instances2, rest2 = pickle_flatten(obj2, Wrapper)
        it = iter(instances1 + instances2 + [Wrapper(4)])  # pyright: ignore[reportUnknownArgumentType]
        assert pickle_unflatten(it, rest1) == obj1  # pyright: ignore[reportUnknownArgumentType]
        assert pickle_unflatten(it, rest2) == obj2  # pyright: ignore[reportUnknownArgumentType]
        assert list(it) == [Wrapper(4)]  # pyright: ignore[reportUnknownArgumentType]

    def test_too_short(self):
        obj = [Wrapper(1), Wrapper(2)]
        instances, rest = pickle_flatten(obj, Wrapper)
        with pytest.raises(ValueError, match="Not enough"):
            pickle_unflatten(instances[:1], rest)  # pyright: ignore[reportUnknownArgumentType]

    def test_recursion(self):
        obj: list[object] = [Wrapper(1)]
        obj.append(obj)

        instances, rest = pickle_flatten(obj, Wrapper)
        assert instances == [Wrapper(1)]

        obj2 = pickle_unflatten(instances, rest)  # pyright: ignore[reportUnknownArgumentType]
        assert len(obj2) == 2
        assert obj2[0] is obj[0]
        assert obj2[1] is obj2


class TestJAXAutoJIT:
    def test_basic(self, jnp: ModuleType):
        @jax_autojit
        def f(x: Array, k: object = False) -> Array:
            return x + 1 if k else x - 1

        # Basic recognition of static_argnames
        xp_assert_equal(f(jnp.asarray([1, 2])), jnp.asarray([0, 1]))
        xp_assert_equal(f(jnp.asarray([1, 2]), False), jnp.asarray([0, 1]))
        xp_assert_equal(f(jnp.asarray([1, 2]), True), jnp.asarray([2, 3]))
        xp_assert_equal(f(jnp.asarray([1, 2]), 1), jnp.asarray([2, 3]))

        # static argument is not an ArrayLike
        xp_assert_equal(f(jnp.asarray([1, 2]), "foo"), jnp.asarray([2, 3]))

        # static argument is not hashable, but serializable
        xp_assert_equal(f(jnp.asarray([1, 2]), ["foo"]), jnp.asarray([2, 3]))

    def test_wrapper(self, jnp: ModuleType):
        @jax_autojit
        def f(w: Wrapper[Array]) -> Wrapper[Array]:
            return Wrapper(w.x + 1)

        inp = Wrapper(jnp.asarray([1, 2]))
        out = f(inp).x
        xp_assert_equal(out, jnp.asarray([2, 3]))

    def test_static_hashable(self, jnp: ModuleType):
        """Static argument/return value is hashable, but not serializable"""

        class C:
            def __reduce__(self) -> object:  # type: ignore[explicit-override,override]  # pyright: ignore[reportIncompatibleMethodOverride,reportImplicitOverride]
                raise Exception()

        @jax_autojit
        def f(x: object) -> object:
            return x

        inp = C()
        out = f(inp)
        assert out is inp

        # Serializable opaque input contains non-serializable object plus array
        inp = Wrapper((C(), jnp.asarray([1, 2])))
        out = f(inp)
        assert isinstance(out, Wrapper)
        assert out.x[0] is inp.x[0]
        assert out.x[1] is not inp.x[1]
        xp_assert_equal(out.x[1], inp.x[1])  # pyright: ignore[reportUnknownArgumentType]

    def test_arraylikes_are_static(self):
        pytest.importorskip("jax")

        @jax_autojit
        def f(x: list[int]) -> list[int]:
            assert isinstance(x, list)
            assert x == [1, 2]
            return [3, 4]

        out = f([1, 2])
        assert isinstance(out, list)
        assert out == [3, 4]
