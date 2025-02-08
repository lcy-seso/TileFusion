#!/bin/bash

# sudo /home/yincao/opt/cuda-12.6/bin/ncu \
#   -f --set full --export tilefusion ./bench_g2s_copy

sudo /home/yincao/opt/cuda-12.6/bin/ncu \
  -f --set full --export cutlass ./bench_g2s_copy
