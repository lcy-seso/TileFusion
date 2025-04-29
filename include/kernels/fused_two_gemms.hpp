// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "cuda_utils.hpp"
#include "kernel_registry.hpp"

namespace tilefusion::kernels {

template <typename InType, typename AccType,                     //
          typename GIteratorA, typename SharedA, typename RegA,  //
          typename SharedALoader, typename RegALoader,           //
          typename GIteratorB, typename SharedB, typename RegB,  //
          typename SharedBLoader, typename RegBLoader,           //
          typename GIteratorC, typename SharedC, typename RegC,  //
          typename SharedCLoader, typename RegCLoader,           //
          typename RegAcc, typename RegAccCast, typename GlobalD,
          typename SharedD, typename RegD, typename RegDHalf,
          typename StoreRegD, typename StoreSharedD, typename ConvertAcc,
          typename ConvertD>
__global__ void ke_fused_two_gemms(const InType* dA, const InType* dB,
                                   const InType* dC, InType* dD, int kM, int kN,
                                   int kK, int kP, int kTM, int kTN, int kTK,
                                   int kTP);

// declare the host function for flash attention
TILEFUSION_EXPORT void fused_two_gemms(const torch::Tensor& A,
                                       const torch::Tensor& B,
                                       const torch::Tensor& C,
                                       torch::Tensor& D);

}  // namespace tilefusion::kernels
