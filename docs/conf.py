import os
import sys
from collections.abc import Mapping

sys.path.insert(0, os.path.abspath("."))

project = "Bonner Lab | Caching"
copyright = "2022, Raj Magesh Gauthaman"
author = "Raj Magesh Gauthaman"
release = "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx.ext.todo",
]

autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "Bonner Lab | Caching"
html_short_title = "Caching"


def linkcode_resolve(domain: str, info: Mapping[str, str]) -> str:
    if domain != "py":
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    return f"https://github.com/BonnerLab/bonner-caching/blob/main/{filename}.py"
