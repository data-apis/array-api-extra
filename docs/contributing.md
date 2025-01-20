# Contributing

Contributions are welcome from any "array-consuming" library contributors who
have found themselves writing private array-agnostic functions in the process of
converting code to consume the standard.

Thanks to [all contributors](contributors.md) so far!

## How to contribute a new function

- [Open an issue](https://github.com/data-apis/array-api-extra/issues/new) to
  propose the new function. You may want to wait for initial feedback on the
  issue before diving into an implementation. Feel free to skip this step if
  there is already an open issue for the function.
- Add the implementation of your function to
  `src/array_api_extra/_lib/_funcs.py`.
  - Ensure that your function includes type annotations and a
    [numpydoc-style docstring](https://numpydoc.readthedocs.io/en/latest/format.html).
  - Add your function to `__all__` at the top of the file.
- Import your function to `src/array_api_extra/__init__.py` and add it to
  `__all__` there.
- Add a test class for your function in `tests/test_funcs.py`.
- Add your function to `docs/api-reference.md`.
- [Make a PR!](https://github.com/data-apis/array-api-extra/pulls)

## How to add delegation to a function

See [the tracker for adding delegation][delegation-tracker].

[delegation-tracker]: https://github.com/data-apis/array-api-extra/issues/100

- If you would like to discuss the task before diving into the implementation,
  click on the three dots next to the function on the tracker issue, and choose
  "Convert to sub-issue".
- Create a function in `src/array_api_extra/_delegation.py` with a signature
  matching the function in `src/array_api_extra/_lib/_funcs.py`, and move the
  docstring to the new function. Leave a one-line docstring in `_funcs.py`,
  pointing to `_delegation.py` to see the full docstring.
- Also move the initial `array_namespace` call and any input validation over to
  the new function.
- Add delegation to backends using the `if _delegate` pattern. See
  `src/array_api_extra/_lib/_backends.py` for the full list of backends we have
  worked with so far.
- After all delegation layers, return the result from the implementation in
  `_funcs`.
- Simplify the signature in `_funcs.py` to remove impossible arguments now that
  it is only called internally via `_delegation`. For example, the `xp`
  parameter can be changed from type `ModuleType | None` to `ModuleType`.
- Don't worry if you are not sure how to do some of the above steps or think you
  might have done something wrong -
  [make a PR!](https://github.com/data-apis/array-api-extra/pulls)

## Development workflow

If you are an experienced contributor to Python packages, feel free to develop
however you feel comfortable! However, if you would like some guidance,
development of array-api-extra is made easy with
[Pixi](https://pixi.sh/latest/):

- [Clone the repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)
  at <https://github.com/data-apis/array-api-extra>.
- `cd array-api-extra`.
- [Install Pixi](https://pixi.sh/latest/#installation).
- To enter a development environment:

```
pixi shell -e dev
```

- To run the tests:

```
pixi run tests
```

- To generate the coverage report:

```
pixi run coverage
```

- To generate and display the coverage report:

```
pixi run open-coverage
```

- To build the docs locally:

```
pixi run docs
```

- To build and preview the docs locally:

```
pixi run open-docs
```

- To install a [pre-commit](https://pre-commit.com) hook:

```
pixi run pre-commit-install
```

- To run the lint suite:

```
pixi run -e lint lint
```

- To enter an interactive Python prompt:

```
pixi run ipython
```

- To run individual parts of the lint suite separately:

```
pixi run -e lint pre-commit
pixi run -e lint pylint
pixi run -e lint mypy
pixi run -e lint pyright
```

Alternative environments are available with a subset of the dependencies and
tasks available in the `dev` environment:

```
pixi shell -e docs
pixi shell -e tests
pixi shell -e tests-backends
pixi shell -e lint
```

If you run on a host with CUDA hardware, you can enable extra tests:

```
pixi shell -e dev-cuda
pixi shell -e tests-cuda
pixi run -e tests-cuda tests
```
