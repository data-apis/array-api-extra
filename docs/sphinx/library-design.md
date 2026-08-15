# Library Design

(scope)=

## Scope

Functions that are in-scope for this library will:

- Implement functionality which does not already exist in the array API
  standard.
- Implement functionality which may be generally useful across various
  libraries.
- Be implemented with static type annotations and
  [numpydoc-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).
- Be tested against array-api-strict and various known array backends.

Functions are implemented purely in terms of the array API standard where
possible. Where functions must use library-specific helpers for libraries
supported by array-api-compat, this will be clearly marked in their API
reference page.

The following kinds of function are also in-scope:

- Functions which implement
  [array API standard extension](https://data-apis.org/array-api/latest/extensions/index.html)
  functions in terms of functions from the base standard.
- Functions which add functionality (e.g. extra parameters) to functions from
  the standard.

Delegation is added for many functions, to use native implementations for the
given array type instead of the array-agnostic implementations, as this may
increase performance.

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
