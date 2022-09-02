import os
import sys

sys.path.insert(0, os.path.abspath("."))

project = "Bonner Lab | Caching"
copyright = "2022, Raj Magesh Gauthaman"
author = "Raj Magesh Gauthaman"
release = "0.1"

extensions = ["sphinx.ext.autodoc", "sphinx.ext.intersphinx"]

autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    # "netCDF4": ("http://unidata.github.io/netcdf4-python", None),
}

exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "Bonner Lab | Caching"
html_short_title = "Caching"
