#!/bin/bash
set -e

# Create build directory
mkdir -p docs/build/html/api/python
mkdir -p docs/build/html/api/cpp
mkdir -p docs/build/xml

# Install documentation dependencies
pip install -r docs/requirements-docs.txt

# Create necessary directories if they don't exist
mkdir -p docs/source/_static
mkdir -p docs/source/_templates
mkdir -p docs/source/api/generated

# Generate Python API documentation with Sphinx
cd docs/source
sphinx-apidoc -o api/generated ../../tilefusion -f
sphinx-build -b html . ../build/html/api/python

# Generate C++ documentation with Doxygen if available
cd ../..
if command -v doxygen &>/dev/null; then
    doxygen Doxyfile
    echo "C++ API documentation is available at docs/build/html/api/cpp/index.html"
else
    echo "Warning: Doxygen not found. Skipping C++ documentation generation."
    echo "To generate C++ documentation, please install Doxygen:"
    echo "  - Ubuntu/Debian: sudo apt-get install doxygen"
    echo "  - macOS: brew install doxygen"
    echo "  - Windows: Download from https://www.doxygen.nl/download.html"
fi

echo "Documentation generated successfully!"
echo "Python API documentation is available at docs/build/html/api/python/index.html"
