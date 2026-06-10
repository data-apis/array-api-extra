1. Update the version in `pyproject.toml`, `meson.build`, and `src/array_api_extra/__init__.py`.
2. Update the lockfile with `pixi update`.
3. Push your changes to a PR, fix any CI failures, and merge to `main`.
4. Cut a release via the GitHub GUI.
5. Update the version to `{next micro version}.dev0` in `pyproject.toml`, `meson.build`, and `src/array_api_extra/__init__.py`.
6. Push your changes to a PR and merge to `main`.
7. Merge the automated PR to `conda-forge/array-api-extra-feedstock`.
