# array-api-extra

```{toctree}
:maxdepth: 2
:hidden:
index.md
api-reference.md
contributing.md
contributors.md
```

This is a library housing "array-agnostic" implementations of functions built on
top of [the Python array API standard](https://data-apis.org/array-api/).

The intended users of this library are "array-consuming" libraries which are
using [array-api-compat](https://data-apis.org/array-api-compat/) to make their
own library's functions array-agnostic. In this library, they will find a set of
tools which provide _extra_ functionality on top of the array API standard,
which other array-consuming libraries in a similar position have found useful
themselves.

(installation)=

## Installation

`array-api-extra` is available
[on PyPI](https://pypi.org/project/array-api-extra/):

```shell
python -m pip install array-api-extra
```

And
[on conda-forge](https://prefix.dev/channels/conda-forge/packages/array-api-extra):

```shell
micromamba install -c https://repo.prefix.dev/conda-forge array-api-extra
# or
pixi add array-api-extra
```

```{warning}
This library currently provides no backwards-compatibility guarantees!
If you require stability, it is recommended to pin `array-api-extra` to
a specific version, or vendor the library inside your own.
```

(vendoring)=

## Vendoring

To vendor the library, clone
[the repository](https://github.com/data-apis/array-api-extra) and copy it into
the appropriate place in your library, like:

```
cp -R array-api-extra/ mylib/vendored/array_api_extra
```

(usage)=

## Usage

Typical usage of this library looks like:

```python
import array_api_extra as xpx

...
xp = array_namespace(x)
y = xp.sum(x)
...
return xpx.atleast_nd(y, ndim=2, xp=xp)
```

```{note}
Functions in this library assume input arrays *are arrays* (not "array-likes") and that
the namespace passed as `xp` is compatible with the standard. This means that
the namespace you pass as `xp` should come from array-api-compat's ``array_namespace``,
or otherwise be compatible with the standard.
```

In the examples shown in the docstrings of functions from this library,
[`array-api-strict`](https://data-apis.org/array-api-strict/) is used as the
array namespace `xp`. In reality, code using this library will be written to
work with any compatible array namespace as `xp`, not any particular
implementation.

(scope)=

## Scope

Functions that are in-scope for this library will:

- Implement functionality which does not already exist in the array API
  standard.
- Implement functionality which may be generally useful across various
  libraries.
- Be implemented purely in terms of the array API standard.
- Be implemented with type annotations and
  [numpydoc-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).
- Be tested against `array-api-strict`.

In particular, the following kinds of function are also in-scope:

- Functions which implement
  [array API standard extension](https://data-apis.org/array-api/2023.12/extensions/index.html)
  functions in terms of functions from the base standard.
- Functions which add functionality (e.g. extra parameters) to functions from
  the standard.

The following features are currently out-of-scope for this library:

- Delegation to known, existing array libraries.
  - It is quite simple to wrap functions in this library to also use existing
    implementations where possible. Such delegation will not live in this
    library for now, but the array-agnostic functions in this library could form
    an array-agnostic backend for such delegating functions in the future, here
    or elsewhere.
- Functions which accept "array-like" input, or standard-incompatible
  namespaces.
  - It is possible to prepare input arrays and a standard-compatible namespace
    via `array-api-compat` downstream in consumer libraries. Avoiding use of
    `array-api-compat` in this library makes it easier to vendor and reduces
    potential redundant calls to `xp.asarray` and `array_namespace`.
  - For proposed alternatives to the `xp=xp` interface, see
    [this issue](https://github.com/data-apis/array-api-extra/issues/6).
- Functions which are specific to a particular domain.
  - These functions may belong better in an array-consuming library which is
    specific to that domain.
