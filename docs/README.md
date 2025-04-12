# Python API Documentation

This directory contains the documentation for the Python API. The documentation is built using [Sphinx](https://www.sphinx-doc.org/) and follows the [Read the Docs](https://readthedocs.org/) theme, generated from docstrings in the source code.

> [!Note]
> Execute commands in this document from the root directory of the project.

## Building Documentation

To build the documentation:

```bash
pip install -r requirements-docs.txt
make docs
```

This will generate HTML documentation in the `build/html` directory.

## Cleaning Documentation

To clean the documentation build files:

```bash
make docs-clean
```

## Previewing Documentation

To preview the documentation using Python's built-in HTTP server:

```bash
python -m http.server --directory docs/build/html
```

Then open your browser and navigate to [http://localhost:8000](http://localhost:8000).

## Documentation Structure

The documentation is organized as follows:

- `source/` - Contains the source files for the documentation
  - `api/` - API reference documentation
  - `conf.py` - Sphinx configuration file
  - `index.rst` - Main documentation index

## Documentation Tools

The documentation uses the following tools:

- [Sphinx](https://www.sphinx-doc.org/) - Documentation generator
- [Read the Docs Theme](https://sphinx-rtd-theme.readthedocs.io/) - Documentation theme
- [Autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) - For automatically generating API documentation
