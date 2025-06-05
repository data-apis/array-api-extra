from types import Any, ModuleType
from typing import TYPE_CHECKING

from ._lib._utils._compat import (
    is_jax_namespace,
    is_torch_namespace,
)
from ._lib._utils._typing import Array, Device, DType

if TYPE_CHECKING:
    import jax
    import torch


class Generator:
    @classmethod
    def create(cls, seed: int, device: Device | None = None) -> "Generator":
        raise NotImplementedError

    def get_state(self) -> Any:
        raise NotImplementedError

    def set_state(self, state: object):
        raise NotImplementedError

    def uniform(
        self,
        shape: tuple[int, ...] = (),
        dtype: DType | None = None,
        minval: float | Array = 0.0,
        maxval: float | Array = 1.0,
    ) -> Array:
        raise NotImplementedError


class JaxGenerator(Generator):
    def __init__(self, key: Array, count: Array | None = None) -> None:
        super().__init__()
        import jax
        import jax.numpy as jnp

        if count is None:
            count = jnp.zeros((), dtype=jnp.uint32)
        else:
            assert isinstance(count, jax.Array)
            assert count.ndim == 0
        assert isinstance(key, jax.Array)
        self._key = key
        self._count = count

    @classmethod
    def create(cls, seed: int, device: Device | None = None) -> "JaxGenerator":
        import jax.random as jr

        key = jr.key(seed).to_device(device)
        return JaxGenerator(key)

    def get_state(self) -> Any:
        import jax.random as jr

        return (jr.key_data(self._key), self._count)

    def set_state(self, state: object):
        import jax
        import jax.random as jr

        assert isinstance(state, tuple)
        key_data, count = state
        assert isinstance(key_data, jax.Array)
        assert isinstance(count, int)
        self._key = jr.wrap_key_data(key_data)
        self._count = count

    def key(self) -> jax.Array:
        """This should be passed to traced functions instead of the generator."""
        import jax.random as jr

        key = jr.fold_in(self._key, self._count)
        self._count += 1
        return key

    def fork(self, samples: int) -> Array:
        """This should be passed to vmapped functions instead of the generator."""
        import jax.random as jr

        return jr.split(self.key(), samples)

    def uniform(
        self,
        shape: tuple[int, ...] = (),
        dtype: DType | None = None,
        minval: float | Array = 0.0,
        maxval: float | Array = 1.0,
    ) -> Array:
        import jax
        import jax.random as jr

        if dtype is None:
            dtype = float
        assert isinstance(minval, float | jax.Array)
        assert isinstance(maxval, float | jax.Array)
        return jr.uniform(self.key(), shape, dtype, minval, maxval)


class TorchGenerator(Generator):
    def __init__(self, generator: "torch.Generator") -> None:
        super().__init__()
        self._generator = generator

    @classmethod
    def create(cls, seed: int, device: Device | None = None) -> "TorchGenerator":
        import torch

        device = "cpu" if device is None else device
        generator = torch.Generator(device)
        generator = generator.manual_seed(seed)
        return TorchGenerator(generator)

    def get_state(self) -> Any:
        return self._generator.get_state()

    def set_state(self, state: object):
        import torch
        assert isinstance(state, torch.Tensor)
        self._generator.set_state(state)

    def uniform(
        self,
        shape: tuple[int, ...] = (),
        dtype: DType | None = None,
        minval: float | Array = 0.0,
        maxval: float | Array = 1.0,
    ) -> Array:
        import torch

        u = torch.rand(*shape, generator=self._generator, dtype=dtype)
        return u * (maxval - minval) + minval


def create_generator(
    xp: ModuleType,
    seed: int,
    *,
    device: Device | None = None,
) -> Generator:
    cls = (
        JaxGenerator
        if is_jax_namespace(xp)
        else TorchGenerator
        if is_torch_namespace(xp)
        else None
    )
    if cls is None:
        raise TypeError
    return cls.create(seed, device)
