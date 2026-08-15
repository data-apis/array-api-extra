"""Sphinx config."""

import importlib
import importlib.metadata
import inspect
from pathlib import Path
from typing import Any

from sphinx import addnodes

_SOURCE_REPOSITORY = "https://github.com/data-apis/array-api-extra"
_SOURCE_BRANCH = "main"
_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
_PACKAGE_SOURCE_ROOT = _REPOSITORY_ROOT / "src"

project = "array-api-extra"
copyright = "Consortium for Python Data API Standards"
author = "Consortium for Python Data API Standards"
version = release = importlib.metadata.version("array_api_extra")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

html_theme = "furo"

html_theme_options: dict[str, Any] = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/data-apis/array-api-extra",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,  # noqa: E501
            "class": "",
        },
    ],
    "source_repository": _SOURCE_REPOSITORY,
    "source_branch": _SOURCE_BRANCH,
    "source_directory": "docs/sphinx/",
}

myst_enable_extensions = [
    "colon_fence",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "array-api": ("https://data-apis.org/array-api/draft", None),
    "dask": ("https://docs.dask.org/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
    "pytest": ("https://docs.pytest.org/en/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

templates_path = ["_templates"]

always_document_param_types = True
typehints_document_overloads = False


def _documented_object(
    doctree: Any,
) -> object | None:  # numpydoc ignore=PR01,RT01
    """Return the first Python object described by a generated page."""
    for signature in doctree.findall(addnodes.desc_signature):
        module_name = signature.get("module")
        fullname = signature.get("fullname")
        if not isinstance(module_name, str) or not isinstance(fullname, str):
            continue

        try:
            obj: object = importlib.import_module(module_name)
            for name in fullname.split("."):
                obj = getattr(obj, name)
        except (AttributeError, ImportError):
            continue
        return obj

    return None


def _repository_source_path(
    obj: object,
) -> Path | None:  # numpydoc ignore=PR01,RT01
    """Return an object's source path when it belongs to this package."""
    if not callable(obj):
        return None

    try:
        filename = inspect.getsourcefile(inspect.unwrap(obj))
        if filename is None:
            return None
        source_path = Path(filename).resolve(strict=True)
        return Path("src") / source_path.relative_to(_PACKAGE_SOURCE_ROOT)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _set_generated_source_links(
    app: Any,
    pagename: str,
    templatename: str,
    context: dict[str, Any],
    doctree: Any,
) -> None:  # numpydoc ignore=PR01
    """Set source links for an autosummary-generated page."""
    del app, templatename
    if doctree is None or not pagename.startswith("generated/"):
        return

    obj = _documented_object(doctree)
    source_path = _repository_source_path(obj) if obj is not None else None
    if source_path is None:
        context["page_source_suffix"] = ""
        return

    source_path_url = source_path.as_posix()
    context["theme_source_view_link"] = (
        f"{_SOURCE_REPOSITORY}/blob/{_SOURCE_BRANCH}/{source_path_url}?plain=true"
    )
    context["theme_source_edit_link"] = (
        f"{_SOURCE_REPOSITORY}/edit/{_SOURCE_BRANCH}/{source_path_url}"
    )


def setup(app: Any) -> dict[str, bool]:  # numpydoc ignore=PR01,RT01
    """Register the generated-page source link handler."""
    app.connect("html-page-context", _set_generated_source_links, priority=900)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
