#!/bin/bash

python3 tests/python/test_scatter_nd.py 2>&1 | tee build.log
