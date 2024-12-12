# Contributing

Contributions are welcome from any "array-consuming" library contributors who
have found themselves writing private array-agnostic functions in the process of
converting code to consume the standard.

Thanks to [all contributors](contributors.md) so far!

## How to contribute a function

- [Open an issue](https://github.com/data-apis/array-api-extra/issues/new) to
  propose the new function.
- Add the implementation of your function to `src/array_api_extra/_funcs.py`.
  - Ensure that your function includes type annotations and a
    [numpydoc-style docstring](https://numpydoc.readthedocs.io/en/latest/format.html).
  - Add your function to `__all__` at the top of the file.
- Import your function to `src/array_api_extra/__init__.py` and add it to
  `__all__` there.
- Add a test class for your function in `tests/test_funcs.py`.
- Add your function to `docs/api-reference.md`.
- [Make a PR!](https://github.com/data-apis/array-api-extra/pulls)

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
pixi run lint
```

- To enter an interactive Python prompt:

```
pixi run ipython
```

- To run individual parts of the lint suite separately:

```
pixi run pre-commit
pixi run pylint
pixi run mypy
pixi run pyright
```

Alternative environments are available with a subset of the dependencies and
tasks available in the `dev` environment:

```
pixi shell -e docs
pixi shell -e tests
pixi shell -e lint
```
