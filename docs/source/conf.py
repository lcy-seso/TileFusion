"""Configuration for the Sphinx documentation."""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2].absolute()))

from python.__version__ import __version__  # noqa: E402

project = "TileFusion"
version = __version__

if version.endswith("dev"):
    version = f"{version} ({datetime.now().strftime('%Y-%m-%d')})"
release = version

# Add any Sphinx extension module names here
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "__pycache__",
    "generated",
]

# The theme to use for HTML and HTML Help pages
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
}

html_static_path = ["_static"]

autodoc_mock_imports = ["tilefusion"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"

suppress_warnings = ["autodoc.import_object"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
