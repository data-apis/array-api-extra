"""
Public testing utilities.

See also _lib._testing for additional private testing utilities.
"""

# https://github.com/scikit-learn/scikit-learn/pull/27910#issuecomment-2568023972
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from functools import wraps
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypeVar, cast

import pytest

from ._lib._utils._compat import is_dask_namespace, is_jax_namespace

__all__ = ["lazy_xp_function", "patch_lazy_xp_functions"]

if TYPE_CHECKING:
    # TODO move ParamSpec outside TYPE_CHECKING
    # depends on scikit-learn abandoning Python 3.9
    # https://github.com/scikit-learn/scikit-learn/pull/27910#issuecomment-2568023972
    from typing import ParamSpec

    from dask.typing import Graph, Key, SchedulerGetCallable
    from typing_extensions import override

    P = ParamSpec("P")
else:
    SchedulerGetCallable = object

    # Sphinx hacks
    class P:  # pylint: disable=missing-class-docstring
        args: tuple
        kwargs: dict

    def override(func: Callable[P, T]) -> Callable[P, T]:
        return func


T = TypeVar("T")


def lazy_xp_function(  # type: ignore[no-any-explicit]
    func: Callable[..., Any],
    *,
    allow_dask_compute: int = 0,
    jax_jit: bool = True,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
) -> None:  # numpydoc ignore=GL07
    """
    Tag a function to be tested on lazy backends.

    Tag a function, which must be imported in the test module globals, so that when any
    tests defined in the same module are executed with ``xp=jax.numpy`` the function is
    replaced with a jitted version of itself, and when it is executed with
    ``xp=dask.array`` the function will raise if it attempts to materialize the graph.
    This will be later expanded to provide test coverage for other lazy backends.

    In order for the tag to be effective, the test or a fixture must call
    :func:`patch_lazy_xp_functions`.

    Parameters
    ----------
    func : callable
        Function to be tested.
    allow_dask_compute : int, optional
        Number of times `func` is allowed to internally materialize the Dask graph. This
        is typically triggered by ``bool()``, ``float()``, or ``np.asarray()``.

        Set to 1 if you are aware that `func` converts the input parameters to numpy and
        want to let it do so at least for the time being, knowing that it is going to be
        extremely detrimental for performance.

        If a test needs values higher than 1 to pass, it is a canary that the conversion
        to numpy/bool/float is happening multiple times, which translates to multiple
        computations of the whole graph. Short of making the function fully lazy, you
        should at least add explicit calls to ``np.asarray()`` early in the function.
        *Note:* the counter of `allow_dask_compute` resets after each call to `func`, so
        a test function that invokes `func` multiple times should still work with this
        parameter set to 1.

        Default: 0, meaning that `func` must be fully lazy and never materialize the
        graph.
    jax_jit : bool, optional
        Set to True to replace `func` with ``jax.jit(func)`` after calling the
        :func:`patch_lazy_xp_functions` test helper with ``xp=jax.numpy``. Set to False
        if `func` is only compatible with eager (non-jitted) JAX. Default: True.
    static_argnums : int | Sequence[int], optional
        Passed to jax.jit. Positional arguments to treat as static (compile-time
        constant). Default: infer from `static_argnames` using
        `inspect.signature(func)`.
    static_argnames : str | Iterable[str], optional
        Passed to jax.jit. Named arguments to treat as static (compile-time constant).
        Default: infer from `static_argnums` using `inspect.signature(func)`.

    See Also
    --------
    patch_lazy_xp_functions : Companion function to call from the test or fixture.
    jax.jit : JAX function to compile a function for performance.

    Examples
    --------
    In ``test_mymodule.py``::

      from array_api_extra.testing import lazy_xp_function from mymodule import myfunc

      lazy_xp_function(myfunc)

      def test_myfunc(xp):
          a = xp.asarray([1, 2])
          # When xp=jax.numpy, this is the same as `b = jax.jit(myfunc)(a)`
          # When xp=dask.array, crash on compute() or persist()
          b = myfunc(a)

    Notes
    -----
    A test function can circumvent this monkey-patching system by calling `func` as an
    attribute of the original module. You need to sanitize your code to make sure this
    does not happen.

    Example::

      import mymodule from mymodule import myfunc

      lazy_xp_function(myfunc)

      def test_myfunc(xp):
          a = xp.asarray([1, 2]) b = myfunc(a)  # This is jitted when xp=jax.numpy c =
          mymodule.myfunc(a)  # This is not
    """
    func.allow_dask_compute = allow_dask_compute  # type: ignore[attr-defined]  # pyright: ignore[reportFunctionMemberAccess]
    if jax_jit:
        func.lazy_jax_jit_kwargs = {  # type: ignore[attr-defined]  # pyright: ignore[reportFunctionMemberAccess]
            "static_argnums": static_argnums,
            "static_argnames": static_argnames,
        }


def patch_lazy_xp_functions(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, *, xp: ModuleType
) -> None:
    """
    Test lazy execution of functions tagged with :func:`lazy_xp_function`.

    If ``xp==jax.numpy``, search for all functions which have been tagged with
    :func:`lazy_xp_function` in the globals of the module that defines the current test
    and wrap them with :func:`jax.jit`. Unwrap them at the end of the test.

    If ``xp==dask.array``, wrap the functions with a decorator that disables
    ``compute()`` and ``persist()``.

    This function should be typically called by your library's `xp` fixture that runs
    tests on multiple backends::

        @pytest.fixture(params=[numpy, array_api_strict, jax.numpy, dask.array])
        def xp(request, monkeypatch):
            patch_lazy_xp_functions(request, monkeypatch, xp=request.param)
            return request.param

    but it can be otherwise be called by the test itself too.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest fixture, as acquired by the test itself or by one of its fixtures.
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture, as acquired by the test itself or by one of its fixtures.
    xp : module
        Array namespace to be tested.

    See Also
    --------
    lazy_xp_function : Tag a function to be tested on lazy backends.
    pytest.FixtureRequest : `request` test function parameter.
    """
    globals_ = cast("dict[str, Any]", request.module.__dict__)  # type: ignore[no-any-explicit]

    if is_dask_namespace(xp):
        for name, func in globals_.items():
            n = getattr(func, "allow_dask_compute", None)
            if n is not None:
                assert isinstance(n, int)
                wrapped = _allow_dask_compute(func, n)
                monkeypatch.setitem(globals_, name, wrapped)

    elif is_jax_namespace(xp):
        import jax

        for name, func in globals_.items():
            kwargs = cast(  # type: ignore[no-any-explicit]
                "dict[str, Any] | None", getattr(func, "lazy_jax_jit_kwargs", None)
            )
            if kwargs is not None:
                # suppress unused-ignore to run mypy in -e lint as well as -e dev
                wrapped = cast(Callable[..., Any], jax.jit(func, **kwargs))  # type: ignore[no-any-explicit,no-untyped-call,unused-ignore]
                monkeypatch.setitem(globals_, name, wrapped)


class CountingDaskScheduler(SchedulerGetCallable):
    """
    Dask scheduler that counts how many times `dask.compute` is called.

    If the number of times exceeds 'max_count', it raises an error.
    This is a wrapper around Dask's own 'synchronous' scheduler.

    Parameters
    ----------
    max_count : int
        Maximum number of allowed calls to `dask.compute`.
    msg : str
        Assertion to raise when the count exceeds `max_count`.
    """

    count: int
    max_count: int
    msg: str

    def __init__(self, max_count: int, msg: str):  # numpydoc ignore=GL08
        self.count = 0
        self.max_count = max_count
        self.msg = msg

    @override
    def __call__(self, dsk: Graph, keys: Sequence[Key] | Key, **kwargs: Any) -> Any:  # type: ignore[no-any-decorated,no-any-explicit] # numpydoc ignore=GL08
        import dask

        self.count += 1
        # This should yield a nice traceback to the
        # offending line in the user's code
        assert self.count <= self.max_count, self.msg

        return dask.get(dsk, keys, **kwargs)  # type: ignore[attr-defined,no-untyped-call] # pyright: ignore[reportPrivateImportUsage]


def _allow_dask_compute(
    func: Callable[P, T], n: int
) -> Callable[P, T]:  # numpydoc ignore=PR01,RT01
    """
    Wrap `func` to raise if it attempts to call `dask.compute` more than `n` times.
    """
    import dask.config

    func_name = getattr(func, "__name__", str(func))
    n_str = f"only up to {n}" if n else "no"
    msg = (
        f"Called `dask.compute()` or `dask.persist()` {n + 1} times, "
        f"but {n_str} calls are allowed. Set "
        f"`lazy_xp_function({func_name}, allow_dask_compute={n + 1})` "
        "to allow for more (but note that this will harm performance). "
    )

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:  # numpydoc ignore=GL08
        scheduler = CountingDaskScheduler(n, msg)
        with dask.config.set({"scheduler": scheduler}):
            return func(*args, **kwargs)

    return wrapper
