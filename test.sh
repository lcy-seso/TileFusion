#!/bin/bash

python tests/python/test_fused_two_gemms.py 2>&1 | tee test.log
