# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
# MIT License.
# --------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(b2b_gemm_example LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      "${PROJECT_SOURCE_DIR}/../../../cmake")
include(generic)

set(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/../../../3rd-party")
include_directories(${THIRD_PARTY_DIR}/cutlass/include)
include_directories("${PROJECT_SOURCE_DIR}/../../../include")

add_executable(fused_gemms b2b_gemm.cu)
target_link_libraries(fused_gemms ${CUDA_CUBLAS_LIBRARIES})
