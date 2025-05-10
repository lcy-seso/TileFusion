// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cell/mod.hpp"
#include "kernels/common.hpp"
#include "kernels/flash_attention_device.cuh"
#include "kernels/ops.hpp"
#include "types/mod.hpp"

namespace tilefusion::kernels {

template <const int kM, const int kN, const int kK, const int kP>
using FlashAttentionShape = TileShape<kM, kN, kK, kP>;

// kernel wrapper
template <typename InType, typename AccType, typename OutType,
          typename WholeShape, typename CtaTileShape>
__attribute__((global)) void kernel_wrapper(const InType* A, const InType* B,
                                            const InType* C, OutType* D) {
    ke_flash_attention<InType, AccType, OutType, WholeShape, CtaTileShape>(
        A, B, C, D);
}

void flash_attention(const torch::Tensor& Q, const torch::Tensor& K,
                     const torch::Tensor& V, torch::Tensor& O,  //
                     int64_t tile_q, int64_t tile_hidden_qk, int64_t tile_kv,
                     int64_t tile_hidden_v, double softmax_scale, bool causal) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    CHECK_INPUT(O);

    const at::ScalarType dtype = Q.scalar_type();
    TORCH_CHECK(dtype == at::ScalarType::Half && K.scalar_type() == dtype &&
                    V.scalar_type() == dtype && O.scalar_type() == dtype,
                "the inputs and output must be half-precision (fp16).");

    const int64_t batch_size = Q.size(0);
    const int64_t length_q = Q.size(1);
    const int64_t length_kv = K.size(2);
    const int64_t hidden_qk = Q.size(2);
    const int64_t hidden_v = V.size(1);

    std::cout << "batch_size: " << batch_size << std::endl
              << "length_q: " << length_q << std::endl
              << "length_kv: " << length_kv << std::endl
              << "hidden_qk: " << hidden_qk << std::endl
              << "hidden_v: " << hidden_v << std::endl;

    using InType = __half;
    using AccType = float;
    using OutType = __half;

    using WholeShape = TileShape<256, 256, 64, 64>;
    using CtaTileShape = TileShape<128, 128, 64, 64>;

    using Config = FlashAttentionTraits<InType, AccType, OutType, WholeShape,
                                        CtaTileShape>;

    int block_x = ceil_div(length_q, tile_q);
    int block_y = ceil_div(hidden_v, tile_hidden_v);

    dim3 grid(block_x, block_y, batch_size);
    dim3 block(Config::kThreads, 1, 1);

    int shm_input = (tile_q * tile_hidden_qk + tile_hidden_qk * tile_kv +
                     tile_kv * tile_hidden_v);
    int shm_output = tile_q * tile_hidden_v;
    int shm_size = shm_input < shm_output ? shm_output * sizeof(OutType)
                                          : shm_input * sizeof(InType);

    auto kernel = &ke_flash_attention<InType, AccType, OutType, Config>;

    if (shm_size > 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    }

    auto dQ = reinterpret_cast<const InType*>(Q.data_ptr());
    auto dK = reinterpret_cast<const InType*>(K.data_ptr());
    auto dV = reinterpret_cast<const InType*>(V.data_ptr());
    auto dO = reinterpret_cast<OutType*>(O.data_ptr());

    kernel<<<grid, block, shm_size, 0>>>(
        dQ, dK, dV, dO,                                  //
        length_q, length_kv, hidden_qk, hidden_v,        //
        tile_q, tile_kv, tile_hidden_qk, tile_hidden_v,  //
        softmax_scale, causal);

    cudaDeviceSynchronize();
}

}  // namespace tilefusion::kernels
