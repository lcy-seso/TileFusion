"""Configuration for the Sphinx documentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2].absolute()))

project = "TileFusion"

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

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
}

autodoc_mock_imports = ["tilefusion"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"

suppress_warnings = ["autodoc.import_object"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
