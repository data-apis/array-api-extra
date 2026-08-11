# Contributing

Thanks to [all contributors](contributors.md) so far!

## Development workflow

Development of array-api-extra is made easy with [Pixi](https://pixi.sh/latest/):

- [Clone the repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)
  at <https://github.com/data-apis/array-api-extra>.
- `cd array-api-extra`.
- [Install Pixi](https://pixi.sh/latest/#installation).

All development tasks are then available via `pixi run`:

```bash
pixi run tests      # run the tests
pixi run open-docs  # build and preview the docs
pixi run lint       # run the full lint suite
pixi run ipython    # spawn an ipython prompt with array-api-extra installed
pixi run hooks      # install pre-commit hooks
```

```{tip}
Run `pixi task list` for a full list of available tasks.
```

Alternative environments are available for the test tasks:

```bash
pixi run --environment=tests-numpy1 tests    # test with numpy<2 installed
pixi run --environment=tests-backends tests  # test with additional CPU array backends
pixi run --environment=tests-cuda tests      # test with CUDA array backends
```

```{tip}
Run `pixi info` for a full list of environments and their tasks.
```

````{note}
You may also enter an activated developer environment shell,
if you prefer this to the `pixi run` task workflow:

```bash
pixi shell --environment=dev
````

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
  - Ensure that `lazy_xp_function` is called on the function if lazy backends
    are supposed to be tested.
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
  parameter can be changed from type `ArrayNamespace | None` to `ArrayNamespace`.
- Don't worry if you are not sure how to do some of the above steps or think you
  might have done something wrong -
  [make a PR!](https://github.com/data-apis/array-api-extra/pulls)
