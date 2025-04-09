"""Configuration for the Sphinx documentation."""

import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath("../.."))

# Project information
project = "TileFusion"

# Add any Sphinx extension module names here
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
]

# Add any paths that contain templates here
templates_path = ["_templates"]

# List of patterns to exclude
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "__pycache__"]

# The theme to use for HTML and HTML Help pages
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files
html_static_path = ["_static"]

# Autodoc settings
autodoc_mock_imports = ["tilefusion"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Suppress warnings
suppress_warnings = ["autodoc.import_object"]
