# array-api-extra

```{toctree}
:maxdepth: 2
:hidden:
self
api-reference.md
api-lazy.md
contributing.md
contributors.md
```

This is a library housing "array-agnostic" implementations of functions built on
top of [the Python array API standard](https://data-apis.org/array-api/), as
well as delegation to existing implementations for known array library backends.

The intended users of this library are "array-consuming" libraries which are
using [array-api-compat](https://data-apis.org/array-api-compat/) to make their
own library's functions array-agnostic. In this library, they will find a set of
tools which provide _extra_ functionality on top of the array API standard,
which other array-consuming libraries in a similar position have found useful
themselves.

It is currently used by:

- [SciPy](https://github.com/scipy/scipy) — Fundamental algorithms for
  scientific computing.
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) — Machine
  Learning in Python.
- _your library? Let us know!_

(installation)=

## Installation

`array-api-extra` is available
[on PyPI](https://pypi.org/project/array-api-extra/):

```shell
uv add array-api-extra
# or
python -m pip install array-api-extra
```

And
[on conda-forge](https://prefix.dev/channels/conda-forge/packages/array-api-extra):

```shell
pixi add array-api-extra
# or
mamba install array-api-extra
```

```{warning}
This library currently provides no backwards-compatibility guarantees!
If you require stability, it is recommended to pin `array-api-extra` to
a specific version, or vendor the library inside your own.
```

```{note}
This library depends on `array-api-compat`. We aim for compatibility with
the latest released version of array-api-compat, and your mileage may vary
with older or dev versions.
```

(vendoring)=

## Vendoring

To vendor the library, clone
[the array-api-extra repository](https://github.com/data-apis/array-api-extra)
and copy it into the appropriate place in your library, like:

```
cp -a array-api-extra/src/array_api_extra mylib/vendored/
```

You may either add a dependency to array-api-compat in your own project, or
vendor it too:

1. Clone
   [the array-api-compat repository](https://github.com/data-apis/array-api-compat)
   and copy it next to your vendored array-api-extra:

   ```
   cp -a array-api-compat/array_api_compat mylib/vendored/
   ```

2. Create a new hook file which array-api-extra will use instead of the
   top-level array-api-compat if present:

   ```
   echo 'from mylib.vendored.array_api_compat import *' > mylib/vendored/_array_api_compat_vendor.py
   ```

This also allows overriding array-api-compat functions if you so wish. E.g. your
`mylib/vendored/_array_api_compat_vendor.py` could look like this:

```python
from mylib.vendored.array_api_compat import *
from mylib.vendored.array_api_compat import array_namespace as _array_namespace_orig


def array_namespace(*xs, **kwargs):
    import mylib

    if any(isinstance(x, mylib.MyArray) for x in xs):
        return mylib
    else:
        return _array_namespace_orig(*xs, **kwargs)
```

```{tip}
See [an example of this in SciPy][scipy-vendor-example].
```

[scipy-vendor-example]:
  https://github.com/scipy/scipy/blob/main/scipy/_lib/_array_api_compat_vendor.py

## Versioning

This project uses [EffVer](https://jacobtomlinson.dev/effver/). TL;DR:

- To bump from version `x.y.z` to `x.(y+1).0` of array-api-extra, expect that
  you may need to put in some work.
- To bump from version `x.y.z` to `x.y.(z+1)`, any work needed on your side
  should be minimal. Please open an issue if this is not the case!

(usage)=

## Usage

Typical usage of this library looks like:

```python
import array_api_extra as xpx

...
xp = array_namespace(x)
y = xp.sum(x)  # use functions from `xp` as normal
...
return xpx.atleast_nd(y, ndim=2, xp=xp)  # use functions from `xpx`, passing `xp=xp`
```

```{note}
Functions in this library assume input arrays *are arrays* (not "array-likes") and that
the namespace passed as `xp` (if given) is compatible with the standard -
this means that it should come from array-api-compat's `array_namespace`,
or otherwise be compatible with the standard.

Calling functions without providing an `xp` argument means that `array_namespace`
is called internally to determine the namespace.
```

In the examples shown in the docstrings of functions from this library,
[array-api-strict](https://data-apis.org/array-api-strict/) is used as the array
namespace `xp`. In reality, code using this library will be written to work with
any compatible array namespace as `xp`, not any particular implementation.

Some functions may only work with specific array libraries supported by
array-api-compat. This should be clearly indicated in the docs - please open an
issue if this is not the case!

(scope)=

## Scope

Functions that are in-scope for this library will:

- Implement functionality which does not already exist in the array API
  standard.
- Implement functionality which may be generally useful across various
  libraries.
- Be implemented with type annotations and
  [numpydoc-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).
- Be tested against array-api-strict and various existing backends.

Functions are implemented purely in terms of the array API standard where
possible. Where functions must use library-specific helpers for libraries
supported by array-api-compat, this will be clearly marked in their API
reference page.

Delegation is added for some functions to use native implementations for the
given array type, instead of the array-agnostic implementations, as this may
increase performance.

In particular, the following kinds of function are also in-scope:

- Functions which implement
  [array API standard extension](https://data-apis.org/array-api/latest/extensions/index.html)
  functions in terms of functions from the base standard.
- Functions which add functionality (e.g. extra parameters) to functions from
  the standard.

The following features are currently out-of-scope for this library:

- Functions which accept "array-like" input, or standard-incompatible
  namespaces.
  - It is possible to prepare input arrays and a standard-compatible namespace
    via array-api-compat downstream in consumer libraries. The `xp` argument can
    also be omitted to infer the standard-compatible namespace using
    `array_namespace` internally.
- Functions which are specific to a particular domain.
  - These functions may belong better in an array-consuming library which is
    specific to that domain.
