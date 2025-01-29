# Tools for lazy backends

These additional functions are meant to be used to support compatibility with
lazy backends, e.g. Dask or Jax:

```{eval-rst}
.. currentmodule:: array_api_extra
.. autosummary::
    :nosignatures:
    :toctree: generated

    lazy_apply
    lazy_raise
    lazy_wait_on
    lazy_warn
    testing.lazy_xp_function
    testing.patch_lazy_xp_functions
```
