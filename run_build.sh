#!/bin/bash

python setup.py develop 2>&1 | tee build.log

sh test.sh 2>&1 | tee test.log
