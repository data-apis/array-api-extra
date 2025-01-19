"""Update operations for read-only arrays."""

# https://github.com/scikit-learn/scikit-learn/pull/27910#issuecomment-2568023972
from __future__ import annotations

import operator
from collections.abc import Callable
from enum import Enum
from types import ModuleType
from typing import ClassVar, cast

from ._utils._compat import array_namespace, is_jax_array, is_writeable_array
from ._utils._typing import Array, Index


class _AtOp(Enum):
    """Operations for use in `xpx.at`."""

    SET = "set"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    POWER = "power"
    MIN = "min"
    MAX = "max"

    # @override from Python 3.12
    def __str__(self) -> str:  # type: ignore[explicit-override]  # pyright: ignore[reportImplicitOverride]
        """
        Return string representation (useful for pytest logs).

        Returns
        -------
        str
            The operation's name.
        """
        return self.value


_undef = object()


class at:  # pylint: disable=invalid-name  # numpydoc ignore=PR02
    """
    Update operations for read-only arrays.

    This implements ``jax.numpy.ndarray.at`` for all writeable
    backends (those that support ``__setitem__``) and routes
    to the ``.at[]`` method for JAX arrays.

    Parameters
    ----------
    x : array
        Input array.
    idx : index, optional
        Only `array API standard compliant indices
        <https://data-apis.org/array-api/latest/API_specification/indexing.html>`_
        are supported.

        You may use two alternate syntaxes::

          >>> import array_api_extra as xpx
          >>> xpx.at(x, idx).set(value)  # or add(value), etc.
          >>> xpx.at(x)[idx].set(value)

    copy : bool, optional
        None (default)
            The array parameter *may* be modified in place if it is
            possible and beneficial for performance.
            You should not reuse it after calling this function.
        True
            Ensure that the inputs are not modified.
        False
            Ensure that the update operation writes back to the input.
            Raise ``ValueError`` if a copy cannot be avoided.

    xp : array_namespace, optional
        The standard-compatible namespace for `x`. Default: infer.

    Returns
    -------
    Updated input array.

    Warnings
    --------
    (a) When you omit the ``copy`` parameter, you should never reuse the parameter
    array later on; ideally, you should reassign it immediately::

        >>> import array_api_extra as xpx
        >>> x = xpx.at(x, 0).set(2)

    The above best practice pattern ensures that the behaviour won't change depending
    on whether ``x`` is writeable or not, as the original ``x`` object is dereferenced
    as soon as ``xpx.at`` returns; this way there is no risk to accidentally update it
    twice.

    On the reverse, the anti-pattern below must be avoided, as it will result in
    different behaviour on read-only versus writeable arrays::

        >>> x = xp.asarray([0, 0, 0])
        >>> y = xpx.at(x, 0).set(2)
        >>> z = xpx.at(x, 1).set(3)

    In the above example, both calls to ``xpx.at`` update ``x`` in place *if possible*.
    This causes the behaviour to diverge depending on whether ``x`` is writeable or not:

    - If ``x`` is writeable, then after the snippet above you'll have
      ``x == y == z == [2, 3, 0]``
    - If ``x`` is read-only, then you'll end up with
      ``x == [0, 0, 0]``, ``y == [2, 0, 0]`` and ``z == [0, 3, 0]``.

    The correct pattern to use if you want diverging outputs from the same input is
    to enforce copies::

        >>> x = xp.asarray([0, 0, 0])
        >>> y = xpx.at(x, 0).set(2, copy=True)  # Never updates x
        >>> z = xpx.at(x, 1).set(3)  # May or may not update x in place
        >>> del x  # avoid accidental reuse of x as we don't know its state anymore

    (b) The array API standard does not support integer array indices.
    The behaviour of update methods when the index is an array of integers is
    undefined and will vary between backends; this is particularly true when the
    index contains multiple occurrences of the same index, e.g.::

        >>> import numpy as np
        >>> import jax.numpy as jnp
        >>> import array_api_extra as xpx
        >>> xpx.at(np.asarray([123]), np.asarray([0, 0])).add(1)
        array([124])
        >>> xpx.at(jnp.asarray([123]), jnp.asarray([0, 0])).add(1)
        Array([125], dtype=int32)

    See Also
    --------
    jax.numpy.ndarray.at : Equivalent array method in JAX.

    Notes
    -----
    `sparse <https://sparse.pydata.org/>`_, as well as read-only arrays from libraries
    not explicitly covered by ``array-api-compat``, are not supported by update
    methods.

    Examples
    --------
    Given either of these equivalent expressions::

      >>> import array_api_extra as xpx
      >>> x = xpx.at(x)[1].add(2)
      >>> x = xpx.at(x, 1).add(2)

    If x is a JAX array, they are the same as::

      >>> x = x.at[1].add(2)

    If x is a read-only numpy array, they are the same as::

      >>> x = x.copy()
      >>> x[1] += 2

    For other known backends, they are the same as::

      >>> x[1] += 2
    """

    _x: Array
    _idx: Index
    __slots__: ClassVar[tuple[str, ...]] = ("_idx", "_x")

    def __init__(
        self, x: Array, idx: Index = _undef, /
    ) -> None:  # numpydoc ignore=GL08
        self._x = x
        self._idx = idx

    def __getitem__(self, idx: Index, /) -> at:  # numpydoc ignore=PR01,RT01
        """
        Allow for the alternate syntax ``at(x)[start:stop:step]``.

        It looks prettier than ``at(x, slice(start, stop, step))``
        and feels more intuitive coming from the JAX documentation.
        """
        if self._idx is not _undef:
            msg = "Index has already been set"
            raise ValueError(msg)
        return at(self._x, idx)

    def _update_common(
        self,
        at_op: _AtOp,
        y: Array,
        /,
        copy: bool | None,
        xp: ModuleType | None,
    ) -> tuple[Array, None] | tuple[None, Array]:  # numpydoc ignore=PR01
        """
        Perform common prepocessing to all update operations.

        Returns
        -------
        tuple
            If the operation can be resolved by ``at[]``, ``(return value, None)``
            Otherwise, ``(None, preprocessed x)``.
        """
        x, idx = self._x, self._idx

        if idx is _undef:
            msg = (
                "Index has not been set.\n"
                "Usage: either\n"
                "    at(x, idx).set(value)\n"
                "or\n"
                "    at(x)[idx].set(value)\n"
                "(same for all other methods)."
            )
            raise ValueError(msg)

        if copy not in (True, False, None):
            msg = f"copy must be True, False, or None; got {copy!r}"
            raise ValueError(msg)

        if copy is None:
            writeable = is_writeable_array(x)
            copy = not writeable
        elif copy:
            writeable = None
        else:
            writeable = is_writeable_array(x)

        if copy:
            if is_jax_array(x):
                # Use JAX's at[]
                func = cast(Callable[[Array], Array], getattr(x.at[idx], at_op.value))
                return func(y), None
            # Emulate at[] behaviour for non-JAX arrays
            # with a copy followed by an update
            if xp is None:
                xp = array_namespace(x)
            x = xp.asarray(x, copy=True)
            if writeable is False:
                # A copy of a read-only numpy array is writeable
                # Note: this assumes that a copy of a writeable array is writeable
                writeable = None

        if writeable is None:
            writeable = is_writeable_array(x)
        if not writeable:
            # sparse crashes here
            msg = f"Can't update read-only array {x}"
            raise ValueError(msg)

        return None, x

    def set(
        self,
        y: Array,
        /,
        copy: bool | None = None,
        xp: ModuleType | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        """Apply ``x[idx] = y`` and return the update array."""
        res, x = self._update_common(_AtOp.SET, y, copy=copy, xp=xp)
        if res is not None:
            return res
        assert x is not None
        x[self._idx] = y
        return x

    def _iop(
        self,
        at_op: _AtOp,
        elwise_op: Callable[[Array, Array], Array],
        y: Array,
        /,
        copy: bool | None,
        xp: ModuleType | None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        """
        ``x[idx] += y`` or equivalent in-place operation on a subset of x.

        which is the same as saying
            x[idx] = x[idx] + y
        Note that this is not the same as
            operator.iadd(x[idx], y)
        Consider for example when x is a numpy array and idx is a fancy index, which
        triggers a deep copy on __getitem__.
        """
        res, x = self._update_common(at_op, y, copy=copy, xp=xp)
        if res is not None:
            return res
        assert x is not None
        x[self._idx] = elwise_op(x[self._idx], y)
        return x

    def add(
        self,
        y: Array,
        /,
        copy: bool | None = None,
        xp: ModuleType | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        """Apply ``x[idx] += y`` and return the updated array."""

        # Note for this and all other methods based on _iop:
        # operator.iadd and operator.add subtly differ in behaviour, as
        # only iadd will trigger exceptions when y has an incompatible dtype.
        return self._iop(_AtOp.ADD, operator.iadd, y, copy=copy, xp=xp)

    def subtract(
        self,
        y: Array,
        /,
        copy: bool | None = None,
        xp: ModuleType | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        """Apply ``x[idx] -= y`` and return the updated array."""
        return self._iop(_AtOp.SUBTRACT, operator.isub, y, copy=copy, xp=xp)

    def multiply(
        self,
        y: Array,
        /,
        copy: bool | None = None,
        xp: ModuleType | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        """Apply ``x[idx] *= y`` and return the updated array."""
        return self._iop(_AtOp.MULTIPLY, operator.imul, y, copy=copy, xp=xp)

    def divide(
        self,
        y: Array,
        /,
        copy: bool | None = None,
        xp: ModuleType | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        """Apply ``x[idx] /= y`` and return the updated array."""
        return self._iop(_AtOp.DIVIDE, operator.itruediv, y, copy=copy, xp=xp)

    def power(
        self,
        y: Array,
        /,
        copy: bool | None = None,
        xp: ModuleType | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        """Apply ``x[idx] **= y`` and return the updated array."""
        return self._iop(_AtOp.POWER, operator.ipow, y, copy=copy, xp=xp)

    def min(
        self,
        y: Array,
        /,
        copy: bool | None = None,
        xp: ModuleType | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        """Apply ``x[idx] = minimum(x[idx], y)`` and return the updated array."""
        if xp is None:
            xp = array_namespace(self._x)
        y = xp.asarray(y)
        return self._iop(_AtOp.MIN, xp.minimum, y, copy=copy, xp=xp)

    def max(
        self,
        y: Array,
        /,
        copy: bool | None = None,
        xp: ModuleType | None = None,
    ) -> Array:  # numpydoc ignore=PR01,RT01
        """Apply ``x[idx] = maximum(x[idx], y)`` and return the updated array."""
        if xp is None:
            xp = array_namespace(self._x)
        y = xp.asarray(y)
        return self._iop(_AtOp.MAX, xp.maximum, y, copy=copy, xp=xp)
