EXAMPLE_DIR := examples
CPP_UT		?= test_gemm
WITH_TEST ?= ON

BUILD_DIR 	:= build
DYNAMIC_LIB	:= $(BUILD_DIR)/libtilefusion.so

# Documentation build variables
PYTHON ?= python3
SPHINXBUILD ?= sphinx-build
SPHINXOPTS ?=
SOURCEDIR = docs/source
BUILDDIR = build

.PHONY: build example unit_test clean docs docs-clean

build:
	@mkdir -p build
	@cd build && cmake -DWITH_TESTING=$(WITH_TEST) .. && make -j$(proc)

$(DYNAMIC_LIB): build
unit_test_cpp: $(DYNAMIC_LIB)
	@cd $(BUILD_DIR) && ctest -R $(CPP_UT) -V

clean:
	@rm -rf build

docs: docs-clean
	cd docs && make html

docs-clean:
	rm -rf docs/build
