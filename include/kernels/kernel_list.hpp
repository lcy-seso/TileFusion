// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "kernel_registry.hpp"
#include "kernels/mod.hpp"

namespace tilefusion ::kernels {

REGISTER_OP(scatter_nd,
            "scatter_nd(Tensor data, Tensor(a!) updates, Tensor indices) -> ()",
            &scatter_nd);

REGISTER_OP(flash_attention,
            "flash_attention(Tensor Q, Tensor K, Tensor V, Tensor(a!) O, "
            "int m, int n, int k, int p) -> ()",
            &flash_attention);

REGISTER_OP(fused_two_gemms,
            "fused_two_gemms(Tensor A, Tensor B, Tensor C, Tensor(a!) D) ->()",
            &fused_two_gemms);

}  // namespace tilefusion::kernels
